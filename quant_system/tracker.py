"""组合收益跟踪模块。

持续跟踪目标仓位组合的收益率：
- 每次运行记录当前持仓和各标的价格
- 下次运行时计算上期持仓的区间收益
- 如果目标权重刷新，用新权重继续跟踪
- 所有记录持久化到 portfolio_history.csv
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

HISTORY_FILE = "portfolio_history.csv"


@dataclass
class TrackingResult:
    """单次跟踪结果。"""
    period_return: float          # 本期加权收益率
    cumulative_return: float      # 累计收益率（从首次跟踪开始）
    holding_details: pd.DataFrame # 本期各标的收益明细
    history: pd.DataFrame         # 完整历史记录


def _load_history(out_dir: Path) -> pd.DataFrame:
    path = out_dir / HISTORY_FILE
    if path.exists():
        try:
            return pd.read_csv(path, encoding="utf-8-sig")
        except Exception:
            pass
    return pd.DataFrame()


def _save_history(df: pd.DataFrame, out_dir: Path) -> None:
    path = out_dir / HISTORY_FILE
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _get_latest_prices(tickers: list[str], prices: pd.DataFrame) -> dict[str, float]:
    """从行情数据中取各标的最新价格。"""
    result: dict[str, float] = {}
    for t in tickers:
        if t in prices.columns:
            px = prices[t].dropna()
            if not px.empty:
                result[t] = float(px.iloc[-1])
    return result


def update_tracking(
    weights: pd.Series,
    prices: pd.DataFrame,
    out_dir: Path,
) -> TrackingResult | None:
    """
    更新组合收益跟踪。

    逻辑：
    1. 读取历史记录，找到上一期的持仓快照
    2. 用当前最新价格计算上一期持仓的区间收益率
    3. 记录本期新持仓（作为下次计算的基准）
    4. 返回收益跟踪结果
    """
    if weights.empty:
        logger.info("目标权重为空，跳过收益跟踪")
        return None

    today = pd.Timestamp.utcnow().strftime("%Y-%m-%d")
    history = _load_history(out_dir)

    # 当前最新价格
    all_tickers = list(set(weights.index.tolist()))
    current_prices = _get_latest_prices(all_tickers, prices)

    # ── 计算上一期持仓收益 ──
    period_return = 0.0
    holding_details_rows: list[dict] = []

    if not history.empty:
        # 取上一期（最新一批记录，同一个 snapshot_date）
        last_date = history["snapshot_date"].iloc[-1]
        last_snapshot = history[history["snapshot_date"] == last_date].copy()

        for _, row in last_snapshot.iterrows():
            ticker = str(row["ticker"])
            entry_price = float(row["entry_price"])
            weight = float(row["weight"])

            if ticker in current_prices and entry_price > 0:
                cur_price = current_prices[ticker]
                ticker_return = (cur_price / entry_price) - 1.0
            else:
                cur_price = entry_price
                ticker_return = 0.0

            weighted_return = weight * ticker_return
            period_return += weighted_return

            holding_details_rows.append({
                "ticker": ticker,
                "weight": weight,
                "entry_price": entry_price,
                "current_price": cur_price,
                "return_pct": ticker_return,
                "weighted_return": weighted_return,
            })

    holding_details = pd.DataFrame(holding_details_rows) if holding_details_rows else pd.DataFrame()

    # ── 计算累计收益率 ──
    # 累计 = 上一期累计 × (1 + 本期收益)
    if not history.empty:
        prev_cum = history["cumulative_return"].iloc[-1]
        cumulative_return = (1 + prev_cum) * (1 + period_return) - 1.0
    else:
        cumulative_return = 0.0  # 首次运行无历史

    # ── 记录本期持仓快照（供下次计算收益用）──
    new_rows: list[dict] = []
    for ticker, weight in weights.items():
        if ticker in current_prices:
            new_rows.append({
                "snapshot_date": today,
                "ticker": ticker,
                "weight": float(weight),
                "entry_price": current_prices[ticker],
                "period_return": period_return,
                "cumulative_return": cumulative_return,
            })

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        if history.empty:
            history = new_df
        else:
            history = pd.concat([history, new_df], ignore_index=True)
        _save_history(history, out_dir)

    logger.info(
        "收益跟踪更新: 本期收益=%.2f%%, 累计收益=%.2f%%",
        period_return * 100, cumulative_return * 100,
    )

    return TrackingResult(
        period_return=period_return,
        cumulative_return=cumulative_return,
        holding_details=holding_details,
        history=history,
    )


def get_performance_summary(out_dir: Path) -> pd.DataFrame:
    """获取每期收益汇总（每个 snapshot_date 一行）。"""
    history = _load_history(out_dir)
    if history.empty:
        return pd.DataFrame()

    summary = (
        history.groupby("snapshot_date")
        .agg(
            n_holdings=("ticker", "count"),
            period_return=("period_return", "first"),
            cumulative_return=("cumulative_return", "first"),
        )
        .reset_index()
    )
    return summary
