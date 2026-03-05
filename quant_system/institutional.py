"""机构持仓与做空数据模块 —— 基于 yfinance 免费数据。

数据源: Yahoo Finance (yfinance)，免费，无需 API Key。

三个维度:
  1. 机构持仓占比 (institutional_pct): 机构投资者持仓占比
  2. 做空压力 (short_pressure): 做空股数占流通股比例 + 做空比例月度变化
  3. 内部人活动 (insider_activity): 近期内部人买入/卖出信号

输出:
  - institutional_pct: 机构持仓占比 (0~1)
  - short_pct: 做空占流通股比例
  - short_change: 做空比例月度变化（正=做空增加，负=做空减少）
  - short_ratio: 空头回补天数
  - insider_buys: 近90天内部人买入次数
  - insider_sells: 近90天内部人卖出次数
  - inst_score: 综合机构因子分 [-1, 1]
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

CACHE_DIR = Path("cache")


def _fetch_institutional_data(ticker: str) -> dict | None:
    """从 yfinance 获取单只 ticker 的机构/做空/内部人数据。"""
    import yfinance as yf

    if ticker.startswith("^"):
        return None

    try:
        t = yf.Ticker(ticker)
        info = t.info
        if not info or not isinstance(info, dict):
            return None

        result: dict = {}

        # 1. 机构持仓占比
        inst_pct = info.get("heldPercentInstitutions", None)
        if inst_pct is not None and isinstance(inst_pct, (int, float)):
            result["institutional_pct"] = float(inst_pct)
        else:
            result["institutional_pct"] = 0.0

        # 2. 做空数据
        short_pct = info.get("shortPercentOfFloat", None)
        result["short_pct"] = float(short_pct) if short_pct is not None and isinstance(short_pct, (int, float)) else 0.0

        short_ratio = info.get("shortRatio", None)
        result["short_ratio"] = float(short_ratio) if short_ratio is not None and isinstance(short_ratio, (int, float)) else 0.0

        shares_short = info.get("sharesShort", 0) or 0
        shares_short_prior = info.get("sharesShortPriorMonth", 0) or 0
        if shares_short_prior > 0:
            result["short_change"] = (shares_short - shares_short_prior) / shares_short_prior
        else:
            result["short_change"] = 0.0

        # 3. 内部人交易（近90天）
        insider_buys = 0
        insider_sells = 0
        try:
            it = t.insider_transactions
            if it is not None and not it.empty and "Start Date" in it.columns:
                now = pd.Timestamp.utcnow()
                cutoff = now - pd.Timedelta(days=90)
                it["_date"] = pd.to_datetime(it["Start Date"], errors="coerce", utc=True)
                recent = it[it["_date"] >= cutoff]
                if not recent.empty and "Transaction" in recent.columns:
                    txns = recent["Transaction"].str.lower().fillna("")
                    # Purchase/Buy-like
                    insider_buys = txns.str.contains("purchase|buy|exercise", na=False).sum()
                    # Sale/Sell-like (排除 gift)
                    insider_sells = txns.str.contains("sale|sell", na=False).sum()
        except Exception:
            pass
        result["insider_buys"] = int(insider_buys)
        result["insider_sells"] = int(insider_sells)

        # 4. 综合机构因子分 inst_score [-1, 1]
        # 维度1: 机构持仓占比 → 高机构持仓=正面信号 (0~0.8映射到 -0.5~1.0)
        ip = result["institutional_pct"]
        inst_sub = min(1.0, max(-0.5, (ip - 0.3) / 0.5))  # 30%以下偏负，80%以上满分

        # 维度2: 做空压力 → 高做空=负面信号
        sp = result["short_pct"]
        sc = result["short_change"]
        # 做空占比: >10%=-1, <1%=+0.5, 线性插值
        short_sub = max(-1.0, min(0.5, 0.5 - sp / 0.10 * 1.5))
        # 做空变化: 做空大幅增加=负面，做空大幅减少=正面
        short_chg_sub = max(-0.5, min(0.5, -sc * 2.0))

        # 维度3: 内部人活动 → 买入=正面，卖出=负面（但卖出很常见，权重低）
        ib = result["insider_buys"]
        is_ = result["insider_sells"]
        if ib + is_ > 0:
            insider_sub = max(-0.3, min(0.5, (ib - is_ * 0.3) * 0.15))
        else:
            insider_sub = 0.0

        # 综合: 机构持仓40% + 做空压力35% + 做空变化15% + 内部人10%
        inst_score = 0.40 * inst_sub + 0.35 * short_sub + 0.15 * short_chg_sub + 0.10 * insider_sub
        result["inst_score"] = round(inst_score, 4)

        return result

    except Exception as e:
        logger.debug("机构数据获取失败 %s: %s: %s", ticker, type(e).__name__, e)
        return None


def fetch_institutional_data(tickers: list[str]) -> pd.DataFrame:
    """批量获取机构/做空/内部人数据，返回 DataFrame (index=ticker)。

    支持当日缓存。

    Returns:
        DataFrame with columns: institutional_pct, short_pct, short_ratio,
                                short_change, insider_buys, insider_sells, inst_score
    """
    today = pd.Timestamp.utcnow().strftime("%Y%m%d")
    cache_path = CACHE_DIR / f"institutional_{today}.pkl"
    if cache_path.exists():
        try:
            cached = pd.read_pickle(cache_path)
            logger.info("使用今日机构数据缓存 (%d 只)", len(cached))
            return cached
        except Exception:
            pass

    valid_tickers = [t for t in tickers if not t.startswith("^")]
    if not valid_tickers:
        return pd.DataFrame()

    logger.info("开始获取机构/做空数据: %d 只标的", len(valid_tickers))
    results: dict[str, dict] = {}
    success_count = 0
    for i, ticker in enumerate(valid_tickers):
        if i > 0 and i % 5 == 0:
            time.sleep(0.5)

        data = _fetch_institutional_data(ticker)
        if data is not None:
            results[ticker] = data
            success_count += 1

    logger.info("机构/做空数据获取完成: 成功 %d / %d", success_count, len(valid_tickers))

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame.from_dict(results, orient="index")
    df.index.name = "ticker"

    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_pickle(cache_path)
    except Exception:
        pass

    return df
