"""DCF（折现现金流）估值模块。

基于 Anthropic financial-services-plugins DCF 方法论：
  - WACC 估算（CAPM: Rf + β × ERP，含债务成本）
  - 5 年 FCF 预测（历史 FCF → 衰减增长）
  - 终值（永续增长法）
  - 年中惯例折现
  - Bear / Base / Bull 三场景
  - WACC × 永续增长率敏感性矩阵
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── 默认宏观参数 ──────────────────────────────────────
RISK_FREE_RATE = 0.043       # 10Y US Treasury（可通过 .env 覆盖）
EQUITY_RISK_PREMIUM = 0.055  # 长期 ERP
TERMINAL_GROWTH_RATE = 0.025 # 永续增长率（GDP 水平）
TAX_RATE = 0.21              # US 企业所得税
PROJECTION_YEARS = 5


@dataclass
class DCFResult:
    """单只股票的 DCF 估值结果。"""
    ticker: str
    current_price: float
    # 三场景内在价值
    bear_value: float
    base_value: float
    bull_value: float
    # 上涨/下跌空间（base 场景）
    upside_pct: float
    # DCF 评分（0~100）
    dcf_score: float
    # 关键假设
    wacc: float
    terminal_growth: float
    fcf_latest: float
    # 敏感性矩阵（5×5 DataFrame: index=WACC, columns=terminal_g）
    sensitivity: pd.DataFrame | None = None
    diagnostics: str = ""


# ── 核心计算 ──────────────────────────────────────────


def _estimate_wacc(
    beta: float,
    debt_to_equity: float,
    risk_free: float = RISK_FREE_RATE,
    erp: float = EQUITY_RISK_PREMIUM,
    tax_rate: float = TAX_RATE,
    cost_of_debt: float | None = None,
) -> float:
    """CAPM + WACC 估算。

    Ke = Rf + β × ERP
    Kd = cost_of_debt × (1 - tax)  （默认 Rf + 1.5%）
    WACC = E/(E+D) × Ke + D/(E+D) × Kd
    """
    beta = max(min(beta, 3.0), 0.3) if not np.isnan(beta) else 1.0
    ke = risk_free + beta * erp

    if cost_of_debt is None:
        cost_of_debt = risk_free + 0.015
    kd = cost_of_debt * (1 - tax_rate)

    # debt_to_equity 单位可能是百分比（如 50 = 50%）或小数
    dte = debt_to_equity if not np.isnan(debt_to_equity) else 0.0
    if dte > 10:  # 百分比格式
        dte = dte / 100.0
    dte = max(dte, 0.0)

    e_weight = 1.0 / (1.0 + dte)
    d_weight = dte / (1.0 + dte)

    wacc = e_weight * ke + d_weight * kd
    return max(min(wacc, 0.25), 0.04)  # 限制在 4%-25%


def _project_fcf(
    fcf_latest: float,
    revenue_growth: float,
    years: int = PROJECTION_YEARS,
    growth_decay: float = 0.80,
    scenario_mult: float = 1.0,
) -> list[float]:
    """预测未来 N 年 FCF。

    增长率逐年衰减：g_t = g_0 × decay^t × scenario_mult
    - Bear: scenario_mult = 0.6
    - Base: scenario_mult = 1.0
    - Bull: scenario_mult = 1.4
    """
    if fcf_latest <= 0:
        # 负 FCF：假设逐年改善
        base_growth = max(revenue_growth, 0.05) * scenario_mult
        projected = []
        fcf = fcf_latest
        for t in range(years):
            # 负 FCF 逐年减少亏损幅度
            improvement = abs(fcf_latest) * 0.15 * (t + 1) * scenario_mult
            fcf = fcf_latest + improvement
            projected.append(fcf)
        return projected

    g0 = max(min(revenue_growth, 1.0), -0.2) if not np.isnan(revenue_growth) else 0.05
    g0 = g0 * scenario_mult

    projected = []
    fcf = fcf_latest
    for t in range(years):
        g_t = g0 * (growth_decay ** t)
        # 限制增长率不低于永续增长率
        g_t = max(g_t, TERMINAL_GROWTH_RATE)
        fcf = fcf * (1 + g_t)
        projected.append(fcf)
    return projected


def _dcf_value(
    fcf_projections: list[float],
    wacc: float,
    terminal_growth: float,
    shares_outstanding: float,
    net_debt: float = 0.0,
) -> float:
    """计算 DCF 内在价值（每股）。

    使用年中惯例（mid-year convention）。
    终值 = 最终年 FCF × (1 + g) / (WACC - g)
    """
    if wacc <= terminal_growth:
        return 0.0
    if shares_outstanding <= 0:
        return 0.0

    # 折现预测期 FCF（年中惯例：t=0.5, 1.5, 2.5...）
    pv_fcf = 0.0
    for t, fcf in enumerate(fcf_projections):
        discount_period = t + 0.5  # 年中惯例
        pv_fcf += fcf / ((1 + wacc) ** discount_period)

    # 终值
    final_fcf = fcf_projections[-1]
    terminal_value = final_fcf * (1 + terminal_growth) / (wacc - terminal_growth)
    # 终值折现到当前
    n = len(fcf_projections)
    pv_terminal = terminal_value / ((1 + wacc) ** (n - 0.5))

    enterprise_value = pv_fcf + pv_terminal
    equity_value = enterprise_value - net_debt

    return max(equity_value / shares_outstanding, 0.0)


def _build_sensitivity(
    fcf_projections: list[float],
    wacc_center: float,
    tg_center: float,
    shares: float,
    net_debt: float,
) -> pd.DataFrame:
    """WACC × 永续增长率 5×5 敏感性矩阵。"""
    wacc_range = [wacc_center + d for d in [-0.02, -0.01, 0.0, 0.01, 0.02]]
    tg_range = [tg_center + d for d in [-0.01, -0.005, 0.0, 0.005, 0.01]]

    matrix = {}
    for tg in tg_range:
        col_label = f"{tg:.1%}"
        col_vals = []
        for w in wacc_range:
            if w <= tg or w <= 0.01:
                col_vals.append(np.nan)
            else:
                val = _dcf_value(fcf_projections, w, tg, shares, net_debt)
                col_vals.append(round(val, 2))
        matrix[col_label] = col_vals

    index_labels = [f"{w:.1%}" for w in wacc_range]
    return pd.DataFrame(matrix, index=index_labels)


# ── 公开接口 ──────────────────────────────────────────


def compute_dcf(
    ticker: str,
    current_price: float,
    fundamentals: dict,
) -> DCFResult | None:
    """为单只股票计算 DCF 估值。

    Args:
        ticker: 股票代码
        current_price: 当前股价
        fundamentals: 来自 yfinance 的基本面数据字典

    Returns:
        DCFResult 或 None（数据不足时）
    """
    fcf = _safe_float(fundamentals.get("free_cashflow"))
    market_cap = _safe_float(fundamentals.get("market_cap"))
    beta = _safe_float(fundamentals.get("beta"), default=1.0)
    dte = _safe_float(fundamentals.get("debt_to_equity"), default=0.0)
    rev_growth = _safe_float(fundamentals.get("revenue_growth"), default=0.05)

    # 数据不足时跳过
    if np.isnan(fcf) or np.isnan(market_cap) or market_cap <= 0 or current_price <= 0:
        return None

    shares = market_cap / current_price

    # 净债务估算：debt_to_equity × equity ≈ market_cap × dte
    if dte > 10:
        dte_ratio = dte / 100.0
    else:
        dte_ratio = max(dte, 0.0)
    net_debt = market_cap * dte_ratio * 0.3  # 粗略估算（实际应从资产负债表取）

    wacc = _estimate_wacc(beta, dte)

    # 三场景 FCF 预测
    fcf_bear = _project_fcf(fcf, rev_growth, scenario_mult=0.6)
    fcf_base = _project_fcf(fcf, rev_growth, scenario_mult=1.0)
    fcf_bull = _project_fcf(fcf, rev_growth, scenario_mult=1.4)

    tg = TERMINAL_GROWTH_RATE

    bear_value = _dcf_value(fcf_bear, wacc, tg, shares, net_debt)
    base_value = _dcf_value(fcf_base, wacc, tg, shares, net_debt)
    bull_value = _dcf_value(fcf_bull, wacc, tg, shares, net_debt)

    # 上涨空间
    upside = (base_value / current_price - 1) if current_price > 0 else 0.0

    # DCF 评分映射：upside → 0~100
    # -50%以下=5, -20%=25, 0%=50, +20%=65, +50%=80, +100%=95
    dcf_score = float(np.interp(
        upside,
        [-0.5, -0.2, 0.0, 0.2, 0.5, 1.0],
        [5.0, 25.0, 50.0, 65.0, 80.0, 95.0],
    ))

    # 敏感性矩阵
    sensitivity = _build_sensitivity(fcf_base, wacc, tg, shares, net_debt)

    diagnostics = (
        f"FCF={fcf/1e6:.0f}M, WACC={wacc:.1%}, β={beta:.2f}, "
        f"D/E={dte:.0f}%, g={rev_growth:.1%}, "
        f"Bear=${bear_value:.1f}, Base=${base_value:.1f}, Bull=${bull_value:.1f}, "
        f"Upside={upside:+.1%}"
    )

    return DCFResult(
        ticker=ticker,
        current_price=current_price,
        bear_value=bear_value,
        base_value=base_value,
        bull_value=bull_value,
        upside_pct=upside,
        dcf_score=dcf_score,
        wacc=wacc,
        terminal_growth=tg,
        fcf_latest=fcf,
        sensitivity=sensitivity,
        diagnostics=diagnostics,
    )


def batch_dcf(
    tickers: list[str],
    prices: pd.DataFrame,
    fundamentals: pd.DataFrame,
) -> dict[str, DCFResult]:
    """批量计算 DCF 估值。

    Returns:
        {ticker: DCFResult} 字典（跳过数据不足的标的）
    """
    results: dict[str, DCFResult] = {}

    for t in tickers:
        if t not in prices.columns or t not in fundamentals.index:
            continue

        current_price = prices[t].dropna().iloc[-1] if not prices[t].dropna().empty else 0.0
        if current_price <= 0:
            continue

        fund_dict = fundamentals.loc[t].to_dict()
        result = compute_dcf(t, current_price, fund_dict)
        if result is not None:
            results[t] = result
            logger.info("DCF %s: %s", t, result.diagnostics)

    logger.info("DCF 估值完成: %d / %d 只", len(results), len(tickers))
    return results


def dcf_results_to_series(dcf_map: dict[str, DCFResult]) -> pd.Series:
    """将 DCF 结果转为 dcf_score Series，用于评分融合。"""
    return pd.Series({t: r.dcf_score for t, r in dcf_map.items()}, dtype=float)


def dcf_summary_df(dcf_map: dict[str, DCFResult]) -> pd.DataFrame:
    """将 DCF 结果转为汇总 DataFrame，用于报告输出。"""
    rows = []
    for t, r in dcf_map.items():
        rows.append({
            "ticker": t,
            "current_price": round(r.current_price, 2),
            "dcf_bear": round(r.bear_value, 2),
            "dcf_base": round(r.base_value, 2),
            "dcf_bull": round(r.bull_value, 2),
            "upside_pct": round(r.upside_pct * 100, 1),
            "dcf_score": round(r.dcf_score, 1),
            "wacc": round(r.wacc * 100, 1),
            "fcf_M": round(r.fcf_latest / 1e6, 0),
        })
    return pd.DataFrame(rows).set_index("ticker") if rows else pd.DataFrame()


def format_dcf_for_prompt(ticker: str, dcf_map: dict[str, DCFResult]) -> str:
    """将 DCF 结果格式化为 LLM prompt 片段。"""
    if ticker not in dcf_map:
        return ""
    r = dcf_map[ticker]
    return (
        f"【DCF 估值】WACC={r.wacc:.1%}, "
        f"Bear=${r.bear_value:.1f}, Base=${r.base_value:.1f}, Bull=${r.bull_value:.1f}, "
        f"当前价=${r.current_price:.1f}, "
        f"Base上涨空间={r.upside_pct:+.0%}"
    )


def _safe_float(val: object, default: float = np.nan) -> float:
    if val is None:
        return default
    try:
        v = float(val)
        return v if not np.isnan(v) else default
    except (TypeError, ValueError):
        return default
