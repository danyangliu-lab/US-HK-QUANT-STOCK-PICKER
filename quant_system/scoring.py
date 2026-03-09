from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ScoreOutput:
    scores: pd.DataFrame
    diagnostics: pd.DataFrame


# ── 绝对阈值打分（替代小样本 rank）──────────────────────────


def _abs_score(s: pd.Series, thresholds: list[tuple[float, float]], default: float = 50.0) -> pd.Series:
    """
    根据绝对阈值区间**线性插值**打分（拉开分数差距）。
    thresholds: [(阈值, 对应分值), ...] 按阈值升序排列。
    在相邻阈值之间做线性插值，低于最低阈值取 default，高于最高阈值取最高分。
    """
    pts = sorted(thresholds, key=lambda x: x[0])
    thresh_vals = [p[0] for p in pts]
    score_vals = [p[1] for p in pts]
    result = np.interp(s.values, thresh_vals, score_vals)
    result = pd.Series(result, index=s.index)
    # 低于最低阈值的用 default
    result = result.where(s >= thresh_vals[0], default)
    return result.where(s.notna(), default)


def _rank_to_100(s: pd.Series, ascending: bool = True) -> pd.Series:
    """保留 rank 打分备用。"""
    if s.dropna().empty:
        return pd.Series(index=s.index, data=50.0)
    r = s.rank(pct=True, ascending=ascending)
    return (r * 100).fillna(50.0)


# ── 技术特征 ──────────────────────────────────────────


def compute_tech_features(prices: pd.DataFrame, volumes: pd.DataFrame,
                          fundamentals: pd.DataFrame | None = None) -> pd.DataFrame:
    latest = prices.iloc[-1]

    ret_5 = prices.pct_change(5, fill_method=None).iloc[-1]
    ret_20 = prices.pct_change(20, fill_method=None).iloc[-1]
    ret_60 = prices.pct_change(60, fill_method=None).iloc[-1]

    ma20 = prices.rolling(20).mean().iloc[-1]
    ma60 = prices.rolling(60).mean().iloc[-1]
    ma120 = prices.rolling(120).mean().iloc[-1]

    above_ma20 = (latest > ma20).astype(float)
    above_ma60 = (latest > ma60).astype(float)

    # 均线斜率（MA20 近5日变化率）
    ma20_series = prices.rolling(20).mean()
    ma20_slope = ma20_series.pct_change(5, fill_method=None).iloc[-1]

    # 突破信号：昨天在MA60之下，今天站上
    yesterday_below_ma60 = (prices.iloc[-2] <= prices.rolling(60).mean().iloc[-2]).astype(float)
    breakout_ma60 = (above_ma60 * yesterday_below_ma60)

    # 52周高点距离
    high_252 = prices.rolling(252).max().iloc[-1]
    dist_52h = latest / high_252 - 1

    # 从近20日低点反弹幅度（回撤修复）
    low_20 = prices.rolling(20).min().iloc[-1]
    drawdown_recovery = (latest / low_20 - 1).replace([np.inf, -np.inf], np.nan)

    # 波动率
    daily_ret = prices.pct_change(fill_method=None).dropna(how="all")
    vol_20 = daily_ret.rolling(20).std().iloc[-1] * np.sqrt(252)

    # 量比
    vol_ratio = (volumes.iloc[-1] / volumes.rolling(20).mean().iloc[-1]).replace(
        [np.inf, -np.inf], np.nan
    )

    # ── 新增量价因子 ──────────────────────────────────────

    # 量价背离：价格创20日新高但成交量萎缩（量比<1），或价格创新低但放量
    # 正值=健康放量上涨，负值=量价背离（顶背离/底背离）
    price_at_20h = (latest >= prices.rolling(20).max().iloc[-1]).astype(float)
    price_at_20l = (latest <= prices.rolling(20).min().iloc[-1]).astype(float)
    # 量价背离分: 新高+缩量=-1, 新高+放量=+1, 新低+放量=-0.5(恐慌抛售), 正常=0
    pv_divergence = pd.Series(0.0, index=latest.index)
    pv_divergence = pv_divergence.where(~(price_at_20h.astype(bool) & (vol_ratio < 1.0)), -1.0)   # 顶背离
    pv_divergence = pv_divergence.where(~(price_at_20h.astype(bool) & (vol_ratio >= 1.0)), 1.0)    # 放量新高
    pv_divergence = pv_divergence.where(~(price_at_20l.astype(bool) & (vol_ratio >= 1.5)), -0.5)   # 恐慌抛售

    # 波动率调整动量（风险调整后的收益，类似夏普比率思路）
    # ret_20 / vol_20，波动率为0或NaN时结果为NaN，由引擎兜底default
    risk_adj_mom = (ret_20 / vol_20).replace([np.inf, -np.inf], np.nan)

    # 收益率偏度（20日日收益率偏度，正偏=有右尾爆发潜力，负偏=左尾风险）
    ret_skew = daily_ret.rolling(20).apply(
        lambda x: x.skew() if len(x.dropna()) >= 10 else np.nan, raw=False
    ).iloc[-1]

    # 换手率（日成交量 / 估算流通股数）
    # 流通股数 ≈ 市值 / 最新价，如果无基本面数据则换手率为NaN
    turnover = pd.Series(np.nan, index=latest.index)
    if fundamentals is not None:
        for t in latest.index:
            if t in fundamentals.index and pd.notna(fundamentals.at[t, "market_cap"]) and latest[t] > 0:
                shares_outstanding = fundamentals.at[t, "market_cap"] / latest[t]
                if shares_outstanding > 0 and t in volumes.columns:
                    avg_vol_5 = volumes[t].iloc[-5:].mean()
                    turnover[t] = avg_vol_5 / shares_outstanding  # 日均换手率

    return pd.DataFrame({
        "ret_5": ret_5,
        "ret_20": ret_20,
        "ret_60": ret_60,
        "above_ma20": above_ma20,
        "above_ma60": above_ma60,
        "ma20_slope": ma20_slope,
        "breakout_ma60": breakout_ma60,
        "dist_52h": dist_52h,
        "drawdown_recovery": drawdown_recovery,
        "vol_20": vol_20,
        "vol_ratio": vol_ratio,
        "pv_divergence": pv_divergence,
        "risk_adj_mom": risk_adj_mom,
        "ret_skew": ret_skew,
        "turnover": turnover,
    })


# ── 成长引擎（绝对阈值打分）──────────────────────────────


def score_growth_engine(tickers: list[str], tech: pd.DataFrame, f: pd.DataFrame) -> pd.DataFrame:
    idx = [t for t in tickers if t in tech.index]
    if not idx:
        return pd.DataFrame()
    df = tech.loc[idx].copy()
    fd = f.reindex(idx)

    # 趋势动量（35%）── 激进风格，趋势为王；引入风险调整动量和偏度
    trend = (
        0.18 * _abs_score(df["ret_5"], [(-.10, 5), (-0.05, 18), (0.0, 40), (0.02, 58), (0.05, 78), (0.10, 100)], default=5.0)
        + 0.20 * _abs_score(df["ret_20"], [(-.15, 5), (-0.05, 18), (0.0, 38), (0.03, 58), (0.08, 80), (0.15, 100)], default=5.0)
        + 0.15 * _abs_score(df["ret_60"], [(-.20, 5), (-0.10, 15), (0.0, 38), (0.05, 58), (0.15, 80), (0.30, 100)], default=5.0)
        + 0.10 * _abs_score(df["ma20_slope"], [(-0.02, 5), (-0.01, 20), (0.0, 40), (0.005, 68), (0.02, 98)], default=5.0)
        + 0.07 * (df["above_ma20"] * 90 + (1 - df["above_ma20"]) * 10)
        + 0.07 * (df["above_ma60"] * 90 + (1 - df["above_ma60"]) * 10)
        + 0.06 * (df["breakout_ma60"] * 100 + (1 - df["breakout_ma60"]) * 40)
        # 波动率调整动量（风险调整后的收益质量）
        + 0.10 * _abs_score(df["risk_adj_mom"], [(-1.0, 5), (-0.3, 18), (0.0, 38), (0.3, 62), (0.8, 82), (1.5, 100)], default=38.0)
        # 收益率偏度（正偏=右尾爆发潜力，负偏=左尾风险）
        + 0.07 * _abs_score(df["ret_skew"], [(-1.5, 10), (-0.5, 25), (0.0, 50), (0.5, 72), (1.0, 88), (2.0, 100)], default=50.0)
    )

    # 基本面成长（30%）── 核心成长指标
    rev_score = _abs_score(
        fd["revenue_growth"],
        [(-0.2, 5), (-0.1, 20), (0.0, 38), (0.05, 52), (0.2, 72), (0.5, 88), (1.0, 100)],
        default=35.0,
    )
    earn_score = _abs_score(
        fd["earnings_growth"],
        [(-0.3, 5), (-0.2, 15), (0.0, 38), (0.05, 52), (0.2, 72), (0.5, 88), (1.0, 100)],
        default=35.0,
    )
    gm_score = _abs_score(
        fd["gross_margin"],
        [(-0.1, 5), (0.0, 20), (0.1, 40), (0.2, 58), (0.4, 78), (0.6, 95)],
        default=35.0,
    )
    fcf_score = _abs_score(
        fd["free_cashflow"],
        [(-5e8, 5), (-1e8, 20), (0, 45), (1e8, 70), (5e8, 90), (2e9, 100)],
        default=35.0,
    )
    growth = 0.35 * rev_score + 0.20 * earn_score + 0.20 * gm_score + 0.25 * fcf_score

    # 盈利质量（8%）── 新增：ROE + 负债率，衡量盈利的可持续性和财务健康度
    roe_score = _abs_score(
        fd["return_on_equity"],
        [(-0.1, 5), (0.0, 20), (0.05, 38), (0.10, 55), (0.20, 75), (0.30, 90), (0.50, 100)],
        default=40.0,
    )
    # 负债率：低负债更健康（debt_to_equity 单位是百分比，如 50 = 50%）
    dte_score = _abs_score(
        fd["debt_to_equity"],
        [(0, 100), (30, 85), (60, 68), (100, 50), (200, 28), (400, 10)],
        default=50.0,
    )
    quality = 0.60 * roe_score + 0.40 * dte_score

    # 资金行为（15%）── 整合量价背离和换手率
    flow = (
        0.25 * _abs_score(df["vol_ratio"], [(0.3, 5), (0.5, 20), (1.0, 45), (1.5, 72), (2.5, 90), (4.0, 100)], default=5.0)
        + 0.25 * _abs_score(df["dist_52h"], [(-0.50, 5), (-0.30, 25), (-0.15, 52), (-0.05, 80), (0.0, 100)], default=5.0)
        + 0.20 * _abs_score(df["drawdown_recovery"], [(-0.05, 5), (0.0, 25), (0.03, 50), (0.08, 75), (0.15, 98)], default=5.0)
        # 量价背离（-1=顶背离→低分, +1=放量新高→高分）
        + 0.15 * _abs_score(df["pv_divergence"], [(-1.0, 10), (-0.5, 25), (0.0, 50), (0.5, 72), (1.0, 95)], default=50.0)
        # 换手率（适度换手=活跃，过低=流动性差，过高=投机过热）
        + 0.15 * _abs_score(df["turnover"], [(0.001, 15), (0.005, 35), (0.01, 55), (0.02, 75), (0.05, 90), (0.10, 70)], default=45.0)
    )

    # 估值约束（12%）── 激进风格，估值权重大幅降低，容忍高估值成长股
    fpe_score = _abs_score(
        fd["forward_pe"],
        [(10, 100), (25, 78), (40, 58), (60, 40), (100, 22), (150, 8)],
        default=50.0,
    )
    pe_score = _abs_score(
        fd["trailing_pe"],
        [(10, 100), (25, 78), (40, 58), (60, 40), (100, 20), (200, 5)],
        default=50.0,
    )
    ps_score = _abs_score(
        fd["price_to_sales"],
        [(1.0, 100), (5.0, 78), (12.0, 52), (25.0, 30), (50.0, 10)],
        default=50.0,
    )
    valuation = 0.40 * fpe_score + 0.35 * pe_score + 0.25 * ps_score

    total = 0.35 * trend + 0.30 * growth + 0.08 * quality + 0.15 * flow + 0.12 * valuation

    # 软惩罚替代硬过滤（激进风格：降低惩罚力度，容忍更多成长代价）
    penalty = pd.Series(0.0, index=idx)
    penalty += (fd["revenue_growth"].fillna(0) < 0).astype(float) * 5
    penalty += (fd["gross_margin"].fillna(0) < 0).astype(float) * 4
    penalty += (fd["trailing_pe"].fillna(0) > 150).astype(float) * 4
    penalty += (fd["price_to_sales"].fillna(0) > 50).astype(float) * 3
    # 高负债惩罚
    penalty += (fd["debt_to_equity"].fillna(0) > 300).astype(float) * 3
    total = total - penalty

    # 硬过滤仅保留极端情况（NaN 视为数据不足，不做惩罚）
    hard_filter = ~(
        (fd["revenue_growth"].fillna(0) < -0.3)
        & (fd["gross_margin"].fillna(0) < -0.1)
    )

    return pd.DataFrame({
        "engine": "growth",
        "trend_score": trend,
        "fund_score": growth,
        "quality_score": quality,
        "flow_score": flow,
        "valuation_score": valuation,
        "quant_score": total,
        "hard_filter_pass": hard_filter.astype(int),
        "annual_vol": df["vol_20"],
        "penalty": penalty,
    }, index=idx)


# ── 杠杆 ETF 引擎（趋势 + 波动率 + 底层股票基本面/技术面）──────────


# ── 小盘投机引擎（高赔率、拥抱风险、重趋势+爆发力）──────────────


def score_smallcap_engine(tickers: list[str], tech: pd.DataFrame, f: pd.DataFrame) -> pd.DataFrame:
    """
    小盘投机引擎：市值<10亿的小盘股，追求高赔率。
    设计理念：趋势第一、爆发力为王、基本面轻看、估值不看、高波加分。
    """
    idx = [t for t in tickers if t in tech.index]
    if not idx:
        return pd.DataFrame()
    df = tech.loc[idx].copy()
    fd = f.reindex(idx)

    # ── 趋势动量（40%）── 投机趋势为王，引入风险调整动量
    trend = (
        0.22 * _abs_score(df["ret_5"], [(-0.15, 3), (-0.05, 12), (0.0, 30), (0.03, 55), (0.08, 80), (0.15, 100)], default=3.0)
        + 0.20 * _abs_score(df["ret_20"], [(-0.20, 3), (-0.10, 10), (0.0, 28), (0.05, 55), (0.15, 82), (0.30, 100)], default=3.0)
        + 0.15 * _abs_score(df["ret_60"], [(-0.30, 3), (-0.15, 10), (0.0, 28), (0.10, 58), (0.25, 82), (0.50, 100)], default=3.0)
        + 0.10 * _abs_score(df["ma20_slope"], [(-0.03, 3), (-0.01, 15), (0.0, 35), (0.01, 72), (0.03, 100)], default=3.0)
        + 0.08 * (df["above_ma20"] * 95 + (1 - df["above_ma20"]) * 5)
        + 0.07 * (df["breakout_ma60"] * 100 + (1 - df["breakout_ma60"]) * 30)
        # 风险调整动量（小盘股中高风险调整收益更有价值）
        + 0.10 * _abs_score(df["risk_adj_mom"], [(-1.0, 3), (-0.3, 12), (0.0, 30), (0.3, 58), (0.8, 82), (1.5, 100)], default=30.0)
        # 收益率偏度（正偏=右尾爆发机会，投机股核心）
        + 0.08 * _abs_score(df["ret_skew"], [(-1.5, 5), (-0.5, 18), (0.0, 40), (0.5, 65), (1.0, 85), (2.0, 100)], default=40.0)
    )

    # ── 爆发力（20%）── 小盘股核心：短期异动、放量突破、反弹力度
    explosion = (
        0.35 * _abs_score(df["ret_5"], [(-0.05, 5), (0.0, 15), (0.05, 45), (0.10, 72), (0.20, 92), (0.40, 100)], default=5.0)
        + 0.30 * _abs_score(df["vol_ratio"], [(0.3, 3), (0.8, 15), (1.5, 45), (2.5, 72), (4.0, 90), (8.0, 100)], default=3.0)
        + 0.35 * _abs_score(df["drawdown_recovery"], [(-0.05, 3), (0.0, 15), (0.05, 42), (0.10, 68), (0.20, 88), (0.40, 100)], default=3.0)
    )

    # ── 资金异动（20%）── 量价配合、逼近新高、量价背离、换手率
    flow = (
        0.22 * _abs_score(df["vol_ratio"], [(0.3, 3), (0.8, 18), (1.5, 50), (3.0, 80), (5.0, 95), (10.0, 100)], default=3.0)
        + 0.28 * _abs_score(df["dist_52h"], [(-0.60, 3), (-0.40, 15), (-0.20, 40), (-0.08, 68), (-0.02, 90), (0.0, 100)], default=3.0)
        + 0.20 * _abs_score(df["drawdown_recovery"], [(-0.05, 3), (0.0, 18), (0.05, 45), (0.12, 72), (0.25, 92), (0.50, 100)], default=3.0)
        # 量价背离（投机股中量价背离是重要警示信号）
        + 0.15 * _abs_score(df["pv_divergence"], [(-1.0, 5), (-0.5, 20), (0.0, 48), (0.5, 75), (1.0, 98)], default=48.0)
        # 换手率（投机股高换手=活跃，是好信号）
        + 0.15 * _abs_score(df["turnover"], [(0.002, 10), (0.01, 30), (0.03, 55), (0.06, 78), (0.10, 95), (0.20, 80)], default=40.0)
    )

    # ── 基本面（15%）── 只看营收增速和毛利率，投机股不看盈利和现金流
    rev_score = _abs_score(
        fd["revenue_growth"],
        [(-0.3, 3), (-0.1, 15), (0.0, 30), (0.1, 52), (0.3, 72), (0.8, 92), (2.0, 100)],
        default=30.0,
    )
    gm_score = _abs_score(
        fd["gross_margin"],
        [(-0.2, 5), (0.0, 20), (0.1, 38), (0.3, 62), (0.5, 85), (0.7, 100)],
        default=30.0,
    )
    growth = 0.65 * rev_score + 0.35 * gm_score

    total = 0.40 * trend + 0.20 * explosion + 0.20 * flow + 0.20 * growth

    # 波动率奖励：投机就是要高波，高波反而加分（适度）
    vol_bonus = _abs_score(df["vol_20"], [(0.2, 0), (0.4, 2), (0.6, 5), (0.8, 8), (1.0, 10), (1.5, 12)], default=0.0)
    total = total + vol_bonus

    # 极轻微软惩罚：只惩罚严重亏损
    penalty = pd.Series(0.0, index=idx)
    penalty += (fd["revenue_growth"].fillna(0) < -0.3).astype(float) * 3
    penalty += (fd["gross_margin"].fillna(0) < -0.3).astype(float) * 3
    total = total - penalty

    # 不做硬过滤（投机股什么都可能）
    hard_filter = pd.Series(True, index=idx)

    return pd.DataFrame({
        "engine": "smallcap",
        "trend_score": trend,
        "explosion_score": explosion,
        "flow_score": flow,
        "fund_score": growth,
        "valuation_score": 50.0,
        "quant_score": total,
        "hard_filter_pass": hard_filter.astype(int),
        "annual_vol": df["vol_20"],
        "penalty": penalty,
    }, index=idx)


# ── 杠杆 ETF 引擎（原有）──────────────────────────────────


def score_leverage_engine(
    tickers: list[str],
    tech: pd.DataFrame,
    fundamentals: pd.DataFrame | None = None,
    underlying_map: dict[str, str] | None = None,
    index_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    idx = [t for t in tickers if t in tech.index]
    if not idx:
        return pd.DataFrame()
    df = tech.loc[idx].copy()
    umap = underlying_map or {}
    imap = index_map or {}

    # ETF自身趋势（所有杠杆ETF都评估）
    etf_trend = (
        0.30 * _abs_score(df["ret_5"], [(-0.10, 5), (-0.05, 20), (0.0, 45), (0.03, 70), (0.08, 100)], default=5.0)
        + 0.30 * _abs_score(df["ret_20"], [(-0.15, 5), (-0.10, 15), (0.0, 45), (0.05, 72), (0.15, 100)], default=5.0)
        + 0.20 * (df["above_ma20"] * 90 + (1 - df["above_ma20"]) * 10)
        + 0.20 * _abs_score(df["ma20_slope"], [(-0.03, 5), (-0.02, 15), (0.0, 40), (0.005, 62), (0.02, 95)], default=5.0)
    )

    # 波动率控制（激进风格：高波是杠杆ETF常态，仅极端高波轻度惩罚）
    vol_ctrl = _abs_score(df["vol_20"], [(0.20, 90), (0.40, 70), (0.60, 48), (0.80, 28), (1.0, 10)], default=55.0)

    # 底层股票因子（单股杠杆ETF）
    underlying_fund = pd.Series(50.0, index=idx)
    underlying_tech = pd.Series(50.0, index=idx)
    # 参考指数技术面（指数型杠杆ETF）
    index_tech = pd.Series(50.0, index=idx)
    has_underlying = pd.Series(False, index=idx)
    has_index = pd.Series(False, index=idx)

    for etf, stock in umap.items():
        if etf not in idx:
            continue
        has_underlying[etf] = True

        if stock in tech.index:
            st = tech.loc[stock]
            underlying_tech[etf] = (
                0.30 * _abs_score_scalar(st["ret_20"], [(-0.10, 10), (-0.05, 28), (0.0, 48), (0.05, 72), (0.10, 95)])
                + 0.30 * _abs_score_scalar(st["ret_60"], [(-0.15, 5), (-0.10, 22), (0.0, 45), (0.03, 58), (0.10, 78), (0.20, 100)])
                + 0.20 * (85.0 if st["above_ma60"] > 0.5 else 15.0)
                + 0.20 * _abs_score_scalar(st["ma20_slope"], [(-0.02, 10), (-0.01, 25), (0.0, 45), (0.005, 68), (0.015, 95)])
            )

        if fundamentals is not None and stock in fundamentals.index:
            fd = fundamentals.loc[stock]
            rev = _safe_abs_score(fd.get("revenue_growth"), [(-0.1, 5), (0.0, 35), (0.05, 55), (0.2, 80), (0.5, 100)], 35.0)
            earn = _safe_abs_score(fd.get("earnings_growth"), [(-0.2, 5), (0.0, 35), (0.05, 55), (0.2, 80), (0.5, 100)], 35.0)
            gm = _safe_abs_score(fd.get("gross_margin"), [(-0.1, 5), (0.0, 20), (0.1, 40), (0.2, 58), (0.4, 78), (0.6, 95)], 35.0)
            val = _safe_abs_score(fd.get("forward_pe"), [(15, 95), (30, 75), (50, 50), (80, 28), (120, 10)], 50.0)
            underlying_fund[etf] = 0.35 * rev + 0.25 * earn + 0.20 * gm + 0.20 * val

    # 参考指数技术面（指数型杠杆ETF）
    for etf, ref_idx in imap.items():
        if etf not in idx or has_underlying[etf]:
            continue  # 有底层股票的不用指数
        if ref_idx in tech.index:
            has_index[etf] = True
            ri = tech.loc[ref_idx]
            index_tech[etf] = (
                0.25 * _abs_score_scalar(ri["ret_5"], [(-0.05, 10), (-0.03, 25), (0.0, 48), (0.02, 72), (0.05, 95)])
                + 0.25 * _abs_score_scalar(ri["ret_20"], [(-0.10, 5), (-0.05, 22), (0.0, 48), (0.05, 75), (0.10, 100)])
                + 0.20 * _abs_score_scalar(ri["ret_60"], [(-0.15, 5), (-0.10, 20), (0.0, 45), (0.08, 75), (0.15, 100)])
                + 0.15 * (85.0 if ri["above_ma60"] > 0.5 else 15.0)
                + 0.15 * _abs_score_scalar(ri["ma20_slope"], [(-0.02, 10), (-0.01, 25), (0.0, 45), (0.003, 68), (0.01, 95)])
            )

    # 合成总分（激进风格：趋势为主，波动率轻惩罚）
    total = pd.Series(0.0, index=idx)
    for t in idx:
        if has_underlying[t]:
            # 有底层股票：ETF趋势45% + 底层技术面20% + 底层基本面20% + 波动率15%
            total[t] = 0.45 * etf_trend[t] + 0.20 * underlying_tech[t] + 0.20 * underlying_fund[t] + 0.15 * vol_ctrl[t]
        elif has_index[t]:
            # 有参考指数：ETF趋势50% + 指数技术面30% + 波动率20%
            total[t] = 0.50 * etf_trend[t] + 0.30 * index_tech[t] + 0.20 * vol_ctrl[t]
        else:
            # 无任何参考：纯趋势80% + 波动率20%
            total[t] = 0.80 * etf_trend[t] + 0.20 * vol_ctrl[t]

    return pd.DataFrame({
        "engine": "leverage",
        "trend_score": etf_trend,
        "fund_score": underlying_fund,
        "index_score": index_tech,
        "flow_score": vol_ctrl,
        "valuation_score": 50.0,
        "quant_score": total,
        "hard_filter_pass": 1,
        "annual_vol": df["vol_20"],
        "penalty": 0.0,
    }, index=idx)


def _abs_score_scalar(value: float, thresholds: list[tuple[float, float]], default: float = 50.0) -> float:
    """标量版线性插值打分，用于单个值。"""
    if pd.isna(value):
        return default
    pts = sorted(thresholds, key=lambda x: x[0])
    thresh_vals = [p[0] for p in pts]
    score_vals = [p[1] for p in pts]
    if value < thresh_vals[0]:
        return default
    return float(np.interp(value, thresh_vals, score_vals))


def _safe_abs_score(value: object, thresholds: list[tuple[float, float]], default: float = 50.0) -> float:
    """安全提取并打分，处理 None/NaN。"""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    try:
        return _abs_score_scalar(float(value), thresholds, default)
    except (TypeError, ValueError):
        return default


# ── 分数拉伸（对抗多因子均值收敛）──────────────────────────


def _stretch_scores(s: pd.Series, power: float = 1.5) -> pd.Series:
    """
    对 quant_score 做幂次拉伸，拉开中间段密集区的差距。

    原理：将分数归一化到 [0, 1]，做 power 次幂变换，再映射回 [0, 100]。
    power > 1 时，高分与低分的差距被非线性放大：
    - 原始 80 vs 60 差 20 → 变换后差距约 30+
    - 原始 55 vs 50 差 5  → 变换后差距约 8+
    """
    if s.empty or s.dropna().empty:
        return s
    smin, smax = s.min(), s.max()
    if smax <= smin:
        return s
    # 归一化 → 幂变换 → 映射回 0~100
    normed = (s - smin) / (smax - smin)  # [0, 1]
    stretched = normed ** power             # 幂次放大
    return stretched * 100


# ── 合并 ──────────────────────────────────────────────


def merge_scores(*engine_dfs: pd.DataFrame) -> ScoreOutput:
    dfs = [d for d in engine_dfs if not d.empty]
    if not dfs:
        empty = pd.DataFrame()
        return ScoreOutput(scores=empty, diagnostics=empty)

    # 在每个引擎内部独立做幂次拉伸，避免引擎间系统性偏压
    # （成长引擎和杠杆引擎因子分布不同，合并后再拉伸会放大引擎间偏差）
    stretched_dfs = []
    for df in dfs:
        d = df.copy()
        d["quant_score_raw"] = d["quant_score"].copy()
        d["quant_score"] = _stretch_scores(d["quant_score"], power=1.8)
        stretched_dfs.append(d)

    merged = pd.concat(stretched_dfs, axis=0)
    merged = merged.sort_values("quant_score", ascending=False)
    # 去除因 ticker 同时出现在多个引擎列表中导致的重复索引（保留分数最高的一行）
    if merged.index.duplicated().any():
        logger.warning("检测到 %d 个重复 ticker（可能同时出现在多个引擎），去重保留最高分",
                       merged.index.duplicated().sum())
        merged = merged[~merged.index.duplicated(keep="first")]
    diag_cols = [
        "engine", "trend_score", "fund_score", "quality_score", "growth_score", "index_score",
        "explosion_score", "flow_score", "valuation_score", "hard_filter_pass", "annual_vol",
        "penalty", "quant_score_raw",
    ]
    diagnostics = merged[[c for c in diag_cols if c in merged.columns]].copy()
    return ScoreOutput(scores=merged, diagnostics=diagnostics)
