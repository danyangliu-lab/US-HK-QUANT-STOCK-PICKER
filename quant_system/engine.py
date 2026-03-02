from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import StrategyConfig
from .data import download_market_data
from .llm import LLMConfig, batch_event_score
from .news import NewsContext, fetch_news, format_news_for_prompt
from .scoring import compute_tech_features, merge_scores, score_bluechip_engine, score_growth_engine, score_leverage_engine
from .tracker import TrackingResult, update_tracking

logger = logging.getLogger(__name__)


@dataclass
class RunResult:
    signal_table: pd.DataFrame
    weights: pd.Series
    output_dir: Path
    diff_report: pd.DataFrame
    tracking: TrackingResult | None = None


def _apply_llm_fusion(
    df: pd.DataFrame, llm_map: dict[str, dict[str, object]], cfg: StrategyConfig
) -> pd.DataFrame:
    out = df.copy()
    out["event_score"] = out.index.map(lambda t: llm_map.get(t, {}).get("event_score", 0.0))
    out["risk_flag"] = out.index.map(lambda t: llm_map.get(t, {}).get("risk_flag", 0))
    out["llm_reason"] = out.index.map(lambda t: llm_map.get(t, {}).get("reason", ""))

    # LLM event_score 映射：[-1, 1] → [0, 100]，再做幂次拉伸放大区分度
    # 原始映射差距太小（0.15 vs -0.25 只差4分），需要放大
    raw_llm = out["event_score"].clip(-1, 1)
    llm_norm = (raw_llm + 1) / 2  # [-1,1] → [0,1]
    # 幂次拉伸（power=2.0），让正面/负面评价差距更显著
    llm_stretched = llm_norm ** 2.0 * 100  # [0, 100]

    out["final_score"] = (
        cfg.quant_weight * out["quant_score"]
        + cfg.llm_weight * llm_stretched
    )
    out.loc[out["risk_flag"] == 1, "final_score"] -= 10
    return out


def _build_target_weights(signal: pd.DataFrame, cfg: StrategyConfig) -> pd.Series:
    # Top-N + 最低分双重机制
    candidates = signal[
        (signal["hard_filter_pass"] == 1) & (signal["final_score"] >= cfg.min_score_to_hold)
    ].copy()

    if candidates.empty:
        return pd.Series(dtype=float)

    candidates = candidates.sort_values("final_score", ascending=False).head(cfg.top_n_buy)

    candidates["raw"] = np.maximum(candidates["final_score"] - 40, 0)
    # 激进风格：不用波动率反比压低高弹性股票仓位，改用轻度调节
    vol = candidates["annual_vol"].replace(0, np.nan).fillna(0.3)
    vol_adj = np.clip(vol, 0.15, 0.80)  # 限制波动率调节范围
    candidates["raw"] = candidates["raw"] / (0.5 + 0.5 * vol_adj)  # 弱化波动率对仓位的打压
    candidates["raw"] = candidates["raw"].replace([np.inf, -np.inf], np.nan).fillna(0)

    candidates = candidates[candidates["raw"] > 0]
    if candidates.empty:
        return pd.Series(dtype=float)

    w = candidates["raw"] / candidates["raw"].sum()

    # 单票上限
    max_map = candidates["engine"].map({
        "growth": cfg.max_weight_growth,
        "bluechip": cfg.max_weight_bluechip,
        "leverage": cfg.max_weight_leverage,
    }).fillna(cfg.max_weight_growth)
    w = np.minimum(w, max_map)

    # 杠杆ETF总上限
    lev_mask = w.index.isin(cfg.leverage_etf_tickers)
    if lev_mask.any() and w[lev_mask].sum() > cfg.max_leverage_etf_total:
        w.loc[lev_mask] *= cfg.max_leverage_etf_total / w.loc[lev_mask].sum()

    # 行业上限
    sectors = pd.Series({t: cfg.sector_map.get(t, "Other") for t in w.index})
    for sec, sec_idx in sectors.groupby(sectors).groups.items():
        sec_weight = w.loc[list(sec_idx)].sum()
        if sec_weight > cfg.max_industry_weight:
            w.loc[list(sec_idx)] *= cfg.max_industry_weight / sec_weight

    if w.sum() > 0:
        w = w / w.sum()

    return w.sort_values(ascending=False)


def _load_yesterday_signal(out_path: Path) -> pd.DataFrame | None:
    csvs = sorted(out_path.glob("signal_table_*.csv"))
    if len(csvs) < 2:
        return None
    try:
        return pd.read_csv(csvs[-2], index_col=0, encoding="utf-8-sig")
    except Exception:
        return None


def _compute_diff(today: pd.DataFrame, yesterday: pd.DataFrame | None) -> pd.DataFrame:
    if yesterday is None:
        today_copy = today[["engine", "final_score", "action"]].copy()
        today_copy["prev_action"] = "N/A"
        today_copy["prev_score"] = np.nan
        today_copy["score_change"] = np.nan
        today_copy["signal_change"] = "NEW"
        return today_copy

    merged = today[["engine", "final_score", "action"]].join(
        yesterday[["final_score", "action"]].rename(columns={"final_score": "prev_score", "action": "prev_action"}),
        how="left",
    )
    merged["score_change"] = merged["final_score"] - merged["prev_score"]
    merged["signal_change"] = "UNCHANGED"
    merged.loc[merged["action"] != merged["prev_action"], "signal_change"] = (
        merged["prev_action"].fillna("N/A") + " → " + merged["action"]
    )
    merged.loc[merged["prev_action"].isna(), "signal_change"] = "NEW"
    return merged


def run_daily_pipeline(cfg: StrategyConfig, llm_cfg: LLMConfig, out_dir: str = "outputs") -> RunResult:
    logger.info("开始每日流水线")
    md = download_market_data(cfg.all_tickers, lookback_days=cfg.lookback_days)
    if md.prices.empty:
        raise RuntimeError("未获取到有效行情数据，请检查ticker和网络。")

    valid_cols = [c for c in md.prices.columns if md.prices[c].dropna().shape[0] >= cfg.min_price_days]
    filtered_out = [c for c in md.prices.columns if c not in valid_cols]
    if filtered_out:
        logger.warning("以下标的因价格数据不足 %d 天被过滤: %s", cfg.min_price_days, filtered_out)
    prices = md.prices[valid_cols].copy()
    volumes = md.volumes[valid_cols].copy()
    fundamentals = md.fundamentals.reindex(valid_cols)
    logger.info("有效标的: %d / %d", len(valid_cols), len(cfg.all_tickers))

    tech = compute_tech_features(prices, volumes)

    growth = score_growth_engine(cfg.growth_tickers, tech, fundamentals)
    bluechip = score_bluechip_engine(cfg.bluechip_tickers + cfg.hk_bluechip_tickers, tech, fundamentals)
    leverage = score_leverage_engine(cfg.leverage_etf_tickers, tech, fundamentals, cfg.leverage_underlying_map, cfg.leverage_index_map)
    merged = merge_scores(growth, bluechip, leverage).scores

    if merged.empty:
        raise RuntimeError("所有标的均未通过打分，请检查数据。")

    # 构建名称映射（用于新闻搜索和显示）
    name_map = {}
    for t in merged.index:
        if t in fundamentals.index:
            name_map[t] = str(fundamentals.at[t, "short_name"]) if pd.notna(fundamentals.at[t, "short_name"]) else t
        else:
            name_map[t] = t

    # 抓取新闻（LLM 启用时才抓取）
    news_ctx: NewsContext | None = None
    if llm_cfg.enabled:
        news_ctx = fetch_news(
            tickers=merged.index.tolist(),
            ticker_names=name_map,
            sector_desc=cfg.leverage_sector_desc,
        )

    llm_map = batch_event_score(merged.index.tolist(), merged, fundamentals, llm_cfg, cfg.leverage_sector_desc, news_ctx=news_ctx)
    signal = _apply_llm_fusion(merged, llm_map, cfg)

    # 加入中文/英文简称
    signal.insert(0, "name", signal.index.map(name_map))

    # 信号动作：Top-N + 分数
    signal = signal.sort_values("final_score", ascending=False)
    signal["action"] = "HOLD"
    top_n_tickers = signal.head(cfg.top_n_buy).index
    signal.loc[
        signal.index.isin(top_n_tickers) & (signal["final_score"] >= cfg.min_score_to_buy),
        "action",
    ] = "BUY"
    signal.loc[
        (signal["hard_filter_pass"] == 0) | (signal["final_score"] < cfg.min_score_to_hold),
        "action",
    ] = "REDUCE"

    weights = _build_target_weights(signal, cfg)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 历史信号diff
    yesterday = _load_yesterday_signal(out_path)
    diff_report = _compute_diff(signal, yesterday)

    ts = pd.Timestamp.utcnow().strftime("%Y%m%d")
    signal.to_csv(out_path / f"signal_table_{ts}.csv", encoding="utf-8-sig")
    weights.rename("target_weight").to_csv(out_path / f"target_weights_{ts}.csv", encoding="utf-8-sig")
    diff_report.to_csv(out_path / f"diff_report_{ts}.csv", encoding="utf-8-sig")

    # 组合收益跟踪
    tracking = update_tracking(weights, md.prices, out_path)

    logger.info("流水线完成，输出至 %s", out_path.resolve())
    return RunResult(signal_table=signal, weights=weights, output_dir=out_path, diff_report=diff_report, tracking=tracking)
