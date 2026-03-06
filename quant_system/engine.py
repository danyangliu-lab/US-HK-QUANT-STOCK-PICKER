from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import StrategyConfig
from .data import download_market_data
from .dcf import DCFResult, batch_dcf, dcf_results_to_series, format_dcf_for_prompt
from .llm import LLMConfig, batch_event_score, batch_portfolio_advice, portfolio_overall_analysis
from .institutional import fetch_institutional_data
from .news import NewsContext, fetch_news, format_news_for_prompt
from .scoring import compute_tech_features, merge_scores, score_growth_engine, score_leverage_engine, score_smallcap_engine
from .sentiment import fetch_social_sentiment
from .tracker import BENCHMARK_TICKERS, TrackingResult, update_tracking

logger = logging.getLogger(__name__)


@dataclass
class RunResult:
    signal_table: pd.DataFrame
    weights: pd.Series
    output_dir: Path
    diff_report: pd.DataFrame
    tracking: TrackingResult | None = None
    portfolio_report: pd.DataFrame | None = None  # 持仓投资建议
    portfolio_overall: str | None = None           # 持仓整体分析
    dcf_map: dict[str, DCFResult] | None = None    # DCF 估值结果
    report_path: str | None = None                 # 研报文件路径


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
        "leverage": cfg.max_weight_leverage,
        "smallcap": cfg.max_weight_smallcap,
    }).fillna(cfg.max_weight_growth)
    w = np.minimum(w, max_map)

    # 杠杆ETF总上限
    lev_mask = w.index.isin(cfg.leverage_etf_tickers)
    if lev_mask.any() and w[lev_mask].sum() > cfg.max_leverage_etf_total:
        w.loc[lev_mask] *= cfg.max_leverage_etf_total / w.loc[lev_mask].sum()

    # 小盘投机池总上限
    sc_mask = w.index.isin(cfg.smallcap_tickers)
    if sc_mask.any() and w[sc_mask].sum() > cfg.max_smallcap_total:
        w.loc[sc_mask] *= cfg.max_smallcap_total / w.loc[sc_mask].sum()

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


def run_daily_pipeline(cfg: StrategyConfig, llm_cfg: LLMConfig, out_dir: str = "outputs",
                       enable_report: bool = False) -> RunResult:
    logger.info("开始每日流水线")
    md = download_market_data(cfg.all_tickers, lookback_days=cfg.lookback_days)
    if md.prices.empty:
        raise RuntimeError("未获取到有效行情数据，请检查ticker和网络。")

    valid_cols = [c for c in md.prices.columns if md.prices[c].dropna().shape[0] >= cfg.min_price_days]
    # 杠杆ETF、小盘投机池、参考指数/ETF 放宽数据过滤（≥5天即可）
    # 杠杆ETF上市时间短但有底层股票/参考指数因子可弥补；小盘股数据也可能较短
    relaxed_tickers = set(cfg.leverage_etf_tickers + cfg.smallcap_tickers + list(cfg.leverage_index_map.values()))
    relaxed_cols = [
        c for c in md.prices.columns
        if c not in valid_cols and c in relaxed_tickers and md.prices[c].dropna().shape[0] >= 5
    ]
    valid_cols = valid_cols + relaxed_cols
    filtered_out = [c for c in md.prices.columns if c not in valid_cols]
    if filtered_out:
        logger.warning("以下标的因价格数据不足被过滤: %s", filtered_out)
    if relaxed_cols:
        logger.info("以下标的因属于杠杆/小盘池，放宽数据过滤（≥5天）保留: %s", relaxed_cols)
    prices = md.prices[valid_cols].copy()
    volumes = md.volumes[valid_cols].copy()
    fundamentals = md.fundamentals.reindex(valid_cols)
    logger.info("有效标的: %d / %d", len(valid_cols), len(cfg.all_tickers))

    tech = compute_tech_features(prices, volumes, fundamentals)

    growth = score_growth_engine(cfg.watchlist_growth_tickers, tech, fundamentals)
    leverage = score_leverage_engine(cfg.watchlist_leverage_tickers, tech, fundamentals, cfg.leverage_underlying_map, cfg.leverage_index_map)
    smallcap = score_smallcap_engine(cfg.smallcap_tickers, tech, fundamentals)
    merged = merge_scores(growth, leverage, smallcap).scores

    # VIX 情绪因子全局调节：VIX 高时降低风险偏好（适度压分），VIX 低时加分
    if md.vix is not None and not md.vix.empty and not merged.empty:
        vix_latest = md.vix.iloc[-1]
        vix_ma20 = md.vix.iloc[-20:].mean() if len(md.vix) >= 20 else md.vix.mean()
        # VIX 绝对水平调节：VIX>30 恐慌区域扣分，VIX<15 贪婪区域加分
        if vix_latest > 35:
            vix_adj = -5.0
        elif vix_latest > 30:
            vix_adj = -3.0
        elif vix_latest > 25:
            vix_adj = -1.5
        elif vix_latest < 15:
            vix_adj = 2.0
        elif vix_latest < 18:
            vix_adj = 1.0
        else:
            vix_adj = 0.0
        # VIX 相对变化调节：VIX 急升（高于MA20 20%以上）额外扣分
        if vix_ma20 > 0 and vix_latest / vix_ma20 > 1.20:
            vix_adj -= 2.0
        elif vix_ma20 > 0 and vix_latest / vix_ma20 < 0.85:
            vix_adj += 1.0
        if abs(vix_adj) > 0.01:
            merged["quant_score"] = merged["quant_score"] + vix_adj
            logger.info("VIX情绪调节: VIX=%.1f, MA20=%.1f, 调整=%.1f分", vix_latest, vix_ma20, vix_adj)

    # 分析师评级因子（yfinance）：个股级别调节
    sentiment_df = fetch_social_sentiment(merged.index.tolist())
    if not sentiment_df.empty:
        analyst_adj_count = 0
        for t in merged.index:
            if t not in sentiment_df.index:
                continue
            s = sentiment_df.loc[t]
            analyst_count = s.get("analyst_count", 0)
            analyst_score = s.get("analyst_score", 0.0)  # [-1, 1]
            # 覆盖分析师太少（<3人）不做调节
            if analyst_count < 3:
                continue
            # 分析师综合分映射到调节分: [-1,1] → [-4, +4]
            # 覆盖人数加权：分析师越多越可信（10人时权重=1.0）
            confidence = min(analyst_count / 10.0, 1.0)
            adj = analyst_score * 4.0 * confidence
            merged.at[t, "quant_score"] = merged.at[t, "quant_score"] + adj
            analyst_adj_count += 1
        if analyst_adj_count > 0:
            logger.info("分析师评级调节: %d 只标的", analyst_adj_count)
        merged["analyst_rating"] = merged.index.map(
            lambda t: sentiment_df.at[t, "analyst_score"] if t in sentiment_df.index else 0.0
        )

    # 机构持仓 + 做空因子（yfinance）：个股级别调节
    inst_df = fetch_institutional_data(merged.index.tolist())
    if not inst_df.empty:
        inst_adj_count = 0
        for t in merged.index:
            if t not in inst_df.index:
                continue
            s = inst_df.loc[t]
            inst_score = s.get("inst_score", 0.0)  # [-1, 1]
            # 机构因子综合分映射到调节分: [-1,1] → [-3, +3]
            adj = inst_score * 3.0
            if abs(adj) > 0.01:
                merged.at[t, "quant_score"] = merged.at[t, "quant_score"] + adj
                inst_adj_count += 1
        if inst_adj_count > 0:
            logger.info("机构/做空因子调节: %d 只标的", inst_adj_count)
        merged["inst_score"] = merged.index.map(
            lambda t: inst_df.at[t, "inst_score"] if t in inst_df.index else 0.0
        )
        merged["short_pct"] = merged.index.map(
            lambda t: inst_df.at[t, "short_pct"] if t in inst_df.index else 0.0
        )

    if merged.empty:
        raise RuntimeError("所有标的均未通过打分，请检查数据。")

    # ── DCF 估值 ──
    # 为非杠杆 ETF 的个股计算 DCF 内在价值
    dcf_tickers = [t for t in merged.index if t not in cfg.leverage_etf_tickers]
    dcf_map = batch_dcf(dcf_tickers, prices, fundamentals)
    if dcf_map:
        dcf_scores = dcf_results_to_series(dcf_map)
        for t in dcf_scores.index:
            if t in merged.index:
                # DCF 评分作为 quant_score 的轻度调节因子（±5 分）
                dcf_adj = (dcf_scores[t] - 50) * 0.10  # 50 分为中性，偏离越大调节越多
                merged.at[t, "quant_score"] = merged.at[t, "quant_score"] + dcf_adj
        logger.info("DCF 估值调节: %d 只标的", len(dcf_scores))

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

    # 组合收益跟踪（含基准指数对比）
    benchmark_prices: dict[str, float] = {}
    for bm_ticker in BENCHMARK_TICKERS:
        if bm_ticker in md.prices.columns:
            px = md.prices[bm_ticker].dropna()
            if not px.empty:
                benchmark_prices[bm_ticker] = float(px.iloc[-1])
    if not benchmark_prices:
        # 基准指数不在 all_tickers 中，单独用 yfinance 获取最新价
        try:
            import yfinance as yf
            for bm_ticker in BENCHMARK_TICKERS:
                try:
                    t = yf.Ticker(bm_ticker)
                    hist = t.history(period="5d", auto_adjust=True)
                    if hist is not None and not hist.empty and "Close" in hist.columns:
                        benchmark_prices[bm_ticker] = float(hist["Close"].dropna().iloc[-1])
                except Exception as e:
                    logger.warning("基准指数 %s 价格获取失败: %s", bm_ticker, e)
        except ImportError:
            pass
    tracking = update_tracking(weights, md.prices, out_path, benchmark_prices=benchmark_prices)

    # ── 我的持仓：评分 + 投资建议 ──
    # 持仓池用更宽松的数据过滤（至少5天即可），确保所有实际持仓都出现在报告中
    portfolio_report: pd.DataFrame | None = None
    portfolio_overall_text: str | None = None
    if cfg.portfolio_tickers:
        pf_valid_cols = [c for c in md.prices.columns if md.prices[c].dropna().shape[0] >= 5]
        pf_prices = md.prices[pf_valid_cols].copy()
        pf_volumes = md.volumes[pf_valid_cols].copy()
        pf_fundamentals = md.fundamentals.reindex(pf_valid_cols)
        pf_tech = compute_tech_features(pf_prices, pf_volumes, pf_fundamentals)

        # 记录数据不足无法评分的ticker
        pf_missing = [t for t in cfg.portfolio_tickers if t not in pf_tech.index]
        if pf_missing:
            logger.warning("以下持仓标的因价格数据不足被排除: %s", pf_missing)

        # 持仓个股用成长引擎评分
        pf_stock_growth = score_growth_engine(cfg.portfolio_stock_tickers, pf_tech, pf_fundamentals)
        # 持仓杠杆ETF用杠杆引擎评分
        pf_leverage = score_leverage_engine(
            cfg.portfolio_leverage_tickers, pf_tech, pf_fundamentals,
            cfg.leverage_underlying_map, cfg.leverage_index_map,
        )
        pf_merged = merge_scores(pf_stock_growth, pf_leverage).scores

        # 持仓 DCF 估值（补充主池未覆盖的持仓股）
        pf_dcf_tickers = [t for t in cfg.portfolio_stock_tickers if t not in dcf_map]
        if pf_dcf_tickers:
            pf_dcf = batch_dcf(pf_dcf_tickers, pf_prices, pf_fundamentals)
            dcf_map.update(pf_dcf)

        if not pf_merged.empty:
            # 名称映射
            pf_name_map = {}
            for t in pf_merged.index:
                if t in pf_fundamentals.index:
                    pf_name_map[t] = str(pf_fundamentals.at[t, "short_name"]) if pd.notna(pf_fundamentals.at[t, "short_name"]) else t
                else:
                    pf_name_map[t] = t

            # LLM event_score 融合（复用自选池已有结果，仅补充调用未覆盖的持仓股）
            pf_already = {t: llm_map[t] for t in pf_merged.index if t in llm_map}
            pf_need_call = [t for t in pf_merged.index if t not in llm_map]
            if pf_need_call:
                logger.info("持仓池复用自选池LLM结果 %d 只，补充调用 %d 只: %s",
                            len(pf_already), len(pf_need_call), pf_need_call)
                pf_extra = batch_event_score(
                    pf_need_call, pf_merged, pf_fundamentals,
                    llm_cfg, cfg.leverage_sector_desc, news_ctx=news_ctx,
                )
                pf_already.update(pf_extra)
            else:
                logger.info("持仓池全部复用自选池LLM结果（%d 只），无需额外调用", len(pf_already))
            pf_llm_map = pf_already
            pf_signal = _apply_llm_fusion(pf_merged, pf_llm_map, cfg)
            pf_signal.insert(0, "name", pf_signal.index.map(pf_name_map))
            pf_signal = pf_signal.sort_values("final_score", ascending=False)

            # 信号动作
            pf_signal["action"] = "HOLD"
            pf_signal.loc[pf_signal["final_score"] >= cfg.min_score_to_buy, "action"] = "BUY"
            pf_signal.loc[
                (pf_signal["hard_filter_pass"] == 0) | (pf_signal["final_score"] < cfg.min_score_to_hold),
                "action",
            ] = "REDUCE"

            # 调用 LLM 生成投资建议
            advice_map = batch_portfolio_advice(
                pf_signal.index.tolist(), pf_signal, pf_fundamentals,
                llm_cfg, cfg.leverage_sector_desc, news_ctx=news_ctx,
            )
            pf_signal["advice_action"] = pf_signal.index.map(lambda t: advice_map.get(t, {}).get("action", ""))
            pf_signal["advice_confidence"] = pf_signal.index.map(lambda t: advice_map.get(t, {}).get("confidence", 0.0))
            pf_signal["advice_reason"] = pf_signal.index.map(lambda t: advice_map.get(t, {}).get("reason", ""))

            portfolio_report = pf_signal
            pf_signal.to_csv(out_path / f"portfolio_advice_{ts}.csv", encoding="utf-8-sig")
            logger.info("持仓投资建议已保存")

            # 整体组合分析
            portfolio_overall_text = portfolio_overall_analysis(
                pf_signal, pf_fundamentals, llm_cfg,
                cfg.leverage_sector_desc, news_ctx=news_ctx,
            )
            # 保存到文件
            with open(out_path / f"portfolio_overall_{ts}.txt", "w", encoding="utf-8") as f_out:
                f_out.write(portfolio_overall_text)

    # ── 研报生成 ──
    report_path_str: str | None = None
    if enable_report:
        try:
            from .report import generate_report, generate_llm_enhanced_report
            if llm_cfg.enabled:
                rp = generate_llm_enhanced_report(
                    signal, fundamentals, dcf_map,
                    portfolio_report, portfolio_overall_text, weights,
                    llm_cfg, cfg.leverage_sector_desc, news_ctx, out_dir,
                )
            else:
                rp = generate_report(
                    signal, fundamentals, dcf_map,
                    portfolio_report, portfolio_overall_text, weights,
                    llm_cfg, cfg.leverage_sector_desc, news_ctx, out_dir,
                )
            report_path_str = str(rp)
            logger.info("研报已生成: %s", rp)
        except Exception as e:
            logger.warning("研报生成异常: %s", e)

    logger.info("流水线完成，输出至 %s", out_path.resolve())
    return RunResult(
        signal_table=signal, weights=weights, output_dir=out_path,
        diff_report=diff_report, tracking=tracking,
        portfolio_report=portfolio_report,
        portfolio_overall=portfolio_overall_text,
        dcf_map=dcf_map if dcf_map else None,
        report_path=report_path_str,
    )
