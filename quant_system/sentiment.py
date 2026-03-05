"""分析师评级量化模块 —— 基于 yfinance 免费数据。

数据源: Yahoo Finance (yfinance)，免费，无需 API Key。

三个维度:
  1. 评级共识分 (consensus_score): 基于 strongBuy/buy/hold/sell/strongSell 人数加权
  2. 目标价上涨空间 (target_upside): 分析师目标价均值 vs 当前价格
  3. 近期升级信号 (recent_upgrade): 近30天是否有 upgrade 事件

输出指标:
  - consensus_score: 评级共识分 [-1, 1]，1=全部强买，-1=全部强卖
  - target_upside: 目标价隐含上涨空间，如 0.15 = 15% 上涨空间
  - recent_upgrade: 近30天净升级数（upgrade次数 - downgrade次数）
  - analyst_count: 覆盖分析师总数
  - analyst_score: 综合分 [-1, 1]，融合三个维度
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

CACHE_DIR = Path("cache")


def _fetch_analyst_data(ticker: str) -> dict | None:
    """从 yfinance 获取单只 ticker 的分析师评级数据。"""
    import yfinance as yf

    # 指数和特殊标的跳过
    if ticker.startswith("^"):
        return None

    try:
        t = yf.Ticker(ticker)

        result: dict = {}

        # 1. 评级共识分：基于推荐汇总
        rec = t.recommendations
        if rec is not None and not rec.empty:
            # 取最新一期（period=0m）
            latest = rec.iloc[0]
            sb = int(latest.get("strongBuy", 0))
            b = int(latest.get("buy", 0))
            h = int(latest.get("hold", 0))
            s = int(latest.get("sell", 0))
            ss = int(latest.get("strongSell", 0))
            total = sb + b + h + s + ss
            if total > 0:
                # 加权分: strongBuy=+1, buy=+0.5, hold=0, sell=-0.5, strongSell=-1
                weighted = (sb * 1.0 + b * 0.5 + h * 0.0 + s * (-0.5) + ss * (-1.0)) / total
                result["consensus_score"] = round(weighted, 4)
                result["analyst_count"] = total
                result["strong_buy"] = sb
                result["buy"] = b
                result["hold"] = h
                result["sell"] = s
                result["strong_sell"] = ss
            else:
                return None
        else:
            return None

        # 2. 目标价上涨空间
        try:
            apt = t.analyst_price_targets
            if apt and isinstance(apt, dict):
                current = apt.get("current", 0)
                mean_target = apt.get("mean", 0)
                if current and current > 0 and mean_target and mean_target > 0:
                    result["target_upside"] = round((mean_target - current) / current, 4)
                    result["target_price_mean"] = round(mean_target, 2)
                    result["current_price"] = round(current, 2)
                else:
                    result["target_upside"] = 0.0
            else:
                result["target_upside"] = 0.0
        except Exception:
            result["target_upside"] = 0.0

        # 3. 近期升降级
        try:
            ud = t.upgrades_downgrades
            if ud is not None and not ud.empty:
                # 只看近30天
                now = pd.Timestamp.utcnow()
                cutoff = now - pd.Timedelta(days=30)
                if ud.index.dtype == "datetime64[ns, UTC]" or "datetime" in str(ud.index.dtype).lower():
                    recent = ud[ud.index >= cutoff]
                else:
                    try:
                        ud.index = pd.to_datetime(ud.index, utc=True)
                        recent = ud[ud.index >= cutoff]
                    except Exception:
                        recent = ud.head(5)

                if not recent.empty:
                    actions = recent["Action"].str.lower() if "Action" in recent.columns else pd.Series(dtype=str)
                    upgrades = actions.isin(["up", "upgrade", "init"]).sum()
                    downgrades = actions.isin(["down", "downgrade"]).sum()
                    result["recent_upgrade"] = int(upgrades - downgrades)
                else:
                    result["recent_upgrade"] = 0
            else:
                result["recent_upgrade"] = 0
        except Exception:
            result["recent_upgrade"] = 0

        # 4. 综合分 analyst_score: 融合三个维度
        cs = result.get("consensus_score", 0.0)
        tu = result.get("target_upside", 0.0)
        ru = result.get("recent_upgrade", 0)

        # 目标价上涨空间映射到 [-1, 1]: -20%以下=-1, +30%以上=+1
        tu_score = max(-1.0, min(1.0, tu / 0.30))

        # 近期升级映射: 每次upgrade +0.25, downgrade -0.25, 范围 [-1, 1]
        ru_score = max(-1.0, min(1.0, ru * 0.25))

        # 综合: 共识50% + 目标价30% + 近期升降级20%
        analyst_score = 0.50 * cs + 0.30 * tu_score + 0.20 * ru_score
        result["analyst_score"] = round(analyst_score, 4)

        return result

    except Exception as e:
        logger.debug("分析师评级获取失败 %s: %s: %s", ticker, type(e).__name__, e)
        return None


def fetch_social_sentiment(tickers: list[str]) -> pd.DataFrame:
    """批量获取分析师评级数据，返回 DataFrame (index=ticker)。

    兼容旧接口名称，但内部已改为 yfinance 分析师评级。
    支持当日缓存。

    Returns:
        DataFrame with columns: consensus_score, target_upside, recent_upgrade,
                                analyst_count, analyst_score, ...
        缺失 ticker 不在返回结果中。
    """
    today = pd.Timestamp.utcnow().strftime("%Y%m%d")
    cache_path = CACHE_DIR / f"analyst_{today}.pkl"
    if cache_path.exists():
        try:
            cached = pd.read_pickle(cache_path)
            logger.info("使用今日分析师评级缓存 (%d 只)", len(cached))
            return cached
        except Exception:
            pass

    # 过滤掉指数
    valid_tickers = [t for t in tickers if not t.startswith("^")]
    if not valid_tickers:
        return pd.DataFrame()

    logger.info("开始获取分析师评级: %d 只标的", len(valid_tickers))
    results: dict[str, dict] = {}
    success_count = 0
    for i, ticker in enumerate(valid_tickers):
        if i > 0 and i % 5 == 0:
            time.sleep(0.5)  # 每5只暂停一下，避免触发限速

        data = _fetch_analyst_data(ticker)
        if data is not None:
            results[ticker] = data
            success_count += 1

    logger.info("分析师评级获取完成: 成功 %d / %d", success_count, len(valid_tickers))

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame.from_dict(results, orient="index")
    df.index.name = "ticker"

    # 缓存
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_pickle(cache_path)
    except Exception:
        pass

    return df
