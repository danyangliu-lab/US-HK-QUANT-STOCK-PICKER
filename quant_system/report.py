"""格式化研报生成模块。

基于 Anthropic financial-services-plugins 研报框架：
  - Initiating Coverage 结构（投资摘要→公司分析→财务→估值→风险）
  - Earnings Analysis 框架（季度更新）
  - 三模型交叉验证生成内容
  - 输出 Markdown + CSV（DCF 汇总）

研报结构（中文，面向个人投资者）：
  1. 投资摘要（评级、目标价区间、核心逻辑）
  2. 个股深度分析（每只持仓/推荐股）
  3. DCF 估值汇总
  4. 组合整体分析
  5. 风险提示
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# ── 研报 Prompt 模板（融合 Anthropic 插件知识）──────────────

_REPORT_SYSTEM_PROMPT = """\
你是一位资深美股/港股量化研究分析师，擅长撰写机构级投资研究报告。
你的报告风格参考摩根大通、高盛、摩根士丹利的研报格式，但用中文撰写，面向个人投资者。

报告结构要求：
1. **投资摘要**：一句话核心观点，当前组合评级（进攻/均衡/防守），关键数据
2. **重点个股分析**（每只股票 3-5 句话）：
   - 投资评级（买入/持有/减持）
   - 量化评分与 DCF 估值对比
   - 近期催化剂/风险
   - 操作建议
3. **DCF 估值观点**：对 DCF 结果的专业解读，哪些被低估/高估
4. **组合整体评估**：行业集中度、风险敞口、建议调仓方向
5. **风险提示**：宏观、行业、个股层面的关键风险

写作规范：
- 使用简体中文
- 数据引用格式：「营收增速 25.3%」「PE 32.1x」「DCF Base $245.80」
- 每只股票分析控制在 100-200 字
- 整体报告 2000-4000 字
- 观点要明确，不要模棱两可
- 给出具体的价格区间和操作建议
"""

_STOCK_ANALYSIS_PROMPT = """\
请为以下股票撰写简要投资分析（中文，100-200字）：

股票: {ticker} ({name})
行业: {sector}
市值: {market_cap}
量化评分: {quant_score:.1f} (满分100)
最终评分: {final_score:.1f}
系统信号: {action}
{dcf_block}
{advice_block}
{news_block}

基本面数据:
- 营收增速: {rev_growth}
- 毛利率: {gross_margin}
- Forward PE: {forward_pe}
- ROE: {roe}
- 负债率: {debt_to_equity}
- 年化波动率: {annual_vol}

请输出JSON格式：
{{"rating":"买入/持有/减持","target_range":"$XX-$YY","analysis":"100-200字分析","catalysts":"近期催化剂","risks":"关键风险"}}
"""


def generate_report(
    signal_table: pd.DataFrame,
    fundamentals: pd.DataFrame,
    dcf_map: dict | None = None,
    portfolio_report: pd.DataFrame | None = None,
    portfolio_overall: str | None = None,
    weights: pd.Series | None = None,
    llm_cfg: Any | None = None,
    sector_desc_map: dict[str, str] | None = None,
    news_ctx: Any | None = None,
    out_dir: str = "outputs",
) -> Path:
    """生成格式化研报（Markdown）。

    Args:
        signal_table: 自选池评分表
        fundamentals: 基本面数据
        dcf_map: DCF 估值结果 {ticker: DCFResult}
        portfolio_report: 持仓评分与建议
        portfolio_overall: 持仓整体分析文本
        weights: 目标权重
        llm_cfg: LLM 配置（用于调用模型生成分析文本）
        sector_desc_map: 杠杆 ETF 行业描述
        news_ctx: 新闻上下文
        out_dir: 输出目录

    Returns:
        报告文件路径
    """
    ts = datetime.utcnow().strftime("%Y%m%d")
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    sections: list[str] = []

    # ── 标题 ──
    sections.append(f"# 📊 量化投资研报 — {datetime.utcnow().strftime('%Y年%m月%d日')}\n")

    # ── 第一部分：投资摘要 ──
    sections.append(_build_summary_section(signal_table, fundamentals, dcf_map, portfolio_overall, weights))

    # ── 第二部分：持仓个股分析 ──
    if portfolio_report is not None and not portfolio_report.empty:
        sections.append(_build_portfolio_section(
            portfolio_report, fundamentals, dcf_map,
            llm_cfg, sector_desc_map, news_ctx,
        ))

    # ── 第三部分：DCF 估值汇总 ──
    if dcf_map:
        sections.append(_build_dcf_section(dcf_map))

    # ── 第四部分：自选池 Top 推荐 ──
    sections.append(_build_topn_section(signal_table, fundamentals, dcf_map, weights))

    # ── 第五部分：组合整体分析 ──
    if portfolio_overall:
        sections.append(_build_overall_section(portfolio_overall))

    # ── 第六部分：风险提示 ──
    sections.append(_build_risk_section())

    # ── 写入文件 ──
    report_content = "\n\n".join(sections)
    report_path = out_path / f"research_report_{ts}.md"
    report_path.write_text(report_content, encoding="utf-8")

    # DCF 汇总 CSV
    if dcf_map:
        from .dcf import dcf_summary_df
        dcf_df = dcf_summary_df(dcf_map)
        if not dcf_df.empty:
            dcf_df.to_csv(out_path / f"dcf_valuation_{ts}.csv", encoding="utf-8-sig")

    logger.info("研报已生成: %s (%d 字)", report_path, len(report_content))
    return report_path


# ── 各章节构建 ──────────────────────────────────────────


def _build_summary_section(
    signal: pd.DataFrame,
    fundamentals: pd.DataFrame,
    dcf_map: dict | None,
    overall: str | None,
    weights: pd.Series | None,
) -> str:
    lines = ["## 一、投资摘要\n"]

    n_total = len(signal)
    n_buy = (signal["action"] == "BUY").sum() if "action" in signal.columns else 0
    n_reduce = (signal["action"] == "REDUCE").sum() if "action" in signal.columns else 0
    avg_score = signal["final_score"].mean() if "final_score" in signal.columns else 0

    lines.append(f"- **观察池规模**: {n_total} 只标的")
    lines.append(f"- **买入信号**: {n_buy} 只 | **减仓信号**: {n_reduce} 只")
    lines.append(f"- **平均最终评分**: {avg_score:.1f}/100")

    if weights is not None and not weights.empty:
        top3 = weights.head(3)
        top3_str = "、".join(f"{t}({w*100:.1f}%)" for t, w in top3.items())
        lines.append(f"- **前三大推荐仓位**: {top3_str}")

    if dcf_map:
        undervalued = [t for t, r in dcf_map.items() if r.upside_pct > 0.1]
        overvalued = [t for t, r in dcf_map.items() if r.upside_pct < -0.2]
        if undervalued:
            lines.append(f"- **DCF 低估标的** (>10% 上涨空间): {', '.join(undervalued[:5])}")
        if overvalued:
            lines.append(f"- **DCF 高估标的** (>20% 下跌风险): {', '.join(overvalued[:5])}")

    # 从 overall 中提取总仓位建议
    if overall and "总仓位建议" in overall:
        for line in overall.split("\n"):
            if "总仓位建议" in line:
                lines.append(f"- **{line.strip().lstrip('💰').strip()}**")
                break

    return "\n".join(lines)


def _build_portfolio_section(
    pf: pd.DataFrame,
    fundamentals: pd.DataFrame,
    dcf_map: dict | None,
    llm_cfg: Any,
    sector_desc_map: dict[str, str] | None,
    news_ctx: Any,
) -> str:
    lines = ["## 二、持仓个股分析\n"]

    sdmap = sector_desc_map or {}

    for t in pf.index:
        row = pf.loc[t]
        name = str(row.get("name", t))
        engine = str(row.get("engine", ""))
        final_score = float(row.get("final_score", 0))
        quant_score = float(row.get("quant_score", 0))
        action = str(row.get("action", ""))
        advice_action = str(row.get("advice_action", ""))
        advice_reason = str(row.get("advice_reason", ""))

        fund = fundamentals.loc[t] if t in fundamentals.index else pd.Series()
        sector = sdmap.get(t, str(fund.get("sector", "N/A")) if not fund.empty else "N/A")
        market_cap_raw = float(fund.get("market_cap", 0)) if not fund.empty else 0
        market_cap_str = f"{market_cap_raw/1e9:.1f}B" if market_cap_raw > 0 else "N/A"

        vol_raw = row.get("annual_vol", None)
        vol_str = f"{float(vol_raw):.0%}" if vol_raw is not None and pd.notna(vol_raw) else "N/A"

        # DCF 数据
        dcf_block = ""
        if dcf_map and t in dcf_map:
            r = dcf_map[t]
            dcf_block = (
                f"  - DCF: Bear ${r.bear_value:.1f} / Base ${r.base_value:.1f} / "
                f"Bull ${r.bull_value:.1f} (上涨空间 {r.upside_pct:+.0%})"
            )

        # 评级映射
        if advice_action in ("加仓",):
            rating_emoji = "🟢"
        elif advice_action in ("减仓", "清仓"):
            rating_emoji = "🔴"
        else:
            rating_emoji = "🟡"

        lines.append(f"### {rating_emoji} {t} ({name})")
        lines.append(f"- **引擎**: {engine} | **行业**: {sector} | **市值**: {market_cap_str}")
        lines.append(f"- **量化评分**: {quant_score:.1f} | **最终评分**: {final_score:.1f} | **信号**: {action}")
        lines.append(f"- **AI建议**: {advice_action} | **波动率**: {vol_str}")
        if dcf_block:
            lines.append(dcf_block)
        if advice_reason:
            lines.append(f"- **分析**: {advice_reason}")
        lines.append("")

    return "\n".join(lines)


def _build_dcf_section(dcf_map: dict) -> str:
    lines = ["## 三、DCF 估值汇总\n"]
    lines.append("| 股票 | 现价 | Bear | Base | Bull | 上涨空间 | WACC | DCF评分 |")
    lines.append("|------|------|------|------|------|----------|------|---------|")

    for t, r in sorted(dcf_map.items(), key=lambda x: -x[1].upside_pct):
        upside_str = f"{r.upside_pct:+.0%}"
        lines.append(
            f"| {t} | ${r.current_price:.1f} | ${r.bear_value:.1f} | "
            f"${r.base_value:.1f} | ${r.bull_value:.1f} | {upside_str} | "
            f"{r.wacc:.1%} | {r.dcf_score:.0f} |"
        )

    # 找出低估和高估最多的
    if dcf_map:
        most_undervalued = max(dcf_map.items(), key=lambda x: x[1].upside_pct)
        most_overvalued = min(dcf_map.items(), key=lambda x: x[1].upside_pct)
        lines.append("")
        lines.append(f"> **最具上涨空间**: {most_undervalued[0]} ({most_undervalued[1].upside_pct:+.0%})")
        lines.append(f"> **最需警惕**: {most_overvalued[0]} ({most_overvalued[1].upside_pct:+.0%})")

    return "\n".join(lines)


def _build_topn_section(
    signal: pd.DataFrame,
    fundamentals: pd.DataFrame,
    dcf_map: dict | None,
    weights: pd.Series | None,
) -> str:
    lines = ["## 四、自选池 Top 推荐\n"]

    buy_signal = signal[signal["action"] == "BUY"].head(10) if "action" in signal.columns else signal.head(10)

    if buy_signal.empty:
        lines.append("当前无买入信号标的。")
        return "\n".join(lines)

    lines.append("| 排名 | 股票 | 名称 | 引擎 | 最终评分 | 目标仓位 | DCF上涨 |")
    lines.append("|------|------|------|------|----------|----------|---------|")

    for rank, t in enumerate(buy_signal.index, 1):
        row = buy_signal.loc[t]
        name = str(row.get("name", t))
        engine = str(row.get("engine", ""))
        final_score = float(row.get("final_score", 0))
        weight_str = f"{weights[t]*100:.1f}%" if weights is not None and t in weights.index else "—"

        dcf_upside = "—"
        if dcf_map and t in dcf_map:
            dcf_upside = f"{dcf_map[t].upside_pct:+.0%}"

        lines.append(
            f"| {rank} | {t} | {name} | {engine} | {final_score:.1f} | {weight_str} | {dcf_upside} |"
        )

    return "\n".join(lines)


def _build_overall_section(overall: str) -> str:
    lines = ["## 五、组合整体分析\n"]
    lines.append(overall)
    return "\n".join(lines)


def _build_risk_section() -> str:
    return (
        "## 六、风险提示\n\n"
        "1. **宏观风险**: 美联储加息/降息节奏变化、通胀反复、经济衰退可能\n"
        "2. **地缘政治**: 中美关系、俄乌冲突、中东局势可能引发市场剧烈波动\n"
        "3. **杠杆风险**: 杠杆 ETF 存在波动率衰减，不适合长期持有\n"
        "4. **DCF 模型局限**: 估值基于历史 FCF 外推和假设参数，实际可能偏差较大\n"
        "5. **流动性风险**: 小盘股/港股杠杆产品可能存在流动性不足\n"
        "6. **模型风险**: 量化评分系统基于历史数据和规则，无法预测黑天鹅事件\n\n"
        "> **免责声明**: 本报告由量化系统自动生成，仅供个人投资参考，不构成投资建议。\n"
        "> 投资有风险，入市需谨慎。"
    )


# ── LLM 研报增强（可选）──────────────────────────────────


def generate_llm_enhanced_report(
    signal_table: pd.DataFrame,
    fundamentals: pd.DataFrame,
    dcf_map: dict | None,
    portfolio_report: pd.DataFrame | None,
    portfolio_overall: str | None,
    weights: pd.Series | None,
    llm_cfg: Any,
    sector_desc_map: dict[str, str] | None = None,
    news_ctx: Any | None = None,
    out_dir: str = "outputs",
) -> Path:
    """LLM 增强版研报：调用大模型生成更丰富的分析内容。

    在基础研报之上，额外调用 LLM 生成：
    - 每只持仓的深度分析段落
    - 市场环境综述
    - 投资策略建议
    """
    if llm_cfg is None or not getattr(llm_cfg, "enabled", False):
        return generate_report(
            signal_table, fundamentals, dcf_map,
            portfolio_report, portfolio_overall, weights,
            llm_cfg, sector_desc_map, news_ctx, out_dir,
        )

    from .dcf import format_dcf_for_prompt

    ts = datetime.utcnow().strftime("%Y%m%d")
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    sections: list[str] = []
    sections.append(f"# 📊 量化投资研报（AI增强版） — {datetime.utcnow().strftime('%Y年%m月%d日')}\n")

    # 投资摘要
    sections.append(_build_summary_section(signal_table, fundamentals, dcf_map, portfolio_overall, weights))

    # LLM 增强的持仓分析
    if portfolio_report is not None and not portfolio_report.empty:
        logger.info("========== 开始生成 AI 增强研报 ==========")
        enhanced = _llm_enhanced_portfolio(
            portfolio_report, fundamentals, dcf_map,
            llm_cfg, sector_desc_map, news_ctx,
        )
        sections.append(enhanced)

    # DCF 汇总
    if dcf_map:
        sections.append(_build_dcf_section(dcf_map))

    # Top 推荐
    sections.append(_build_topn_section(signal_table, fundamentals, dcf_map, weights))

    # 组合分析
    if portfolio_overall:
        sections.append(_build_overall_section(portfolio_overall))

    # 风险
    sections.append(_build_risk_section())

    report_content = "\n\n".join(sections)
    report_path = out_path / f"research_report_{ts}.md"
    report_path.write_text(report_content, encoding="utf-8")

    if dcf_map:
        from .dcf import dcf_summary_df
        dcf_df = dcf_summary_df(dcf_map)
        if not dcf_df.empty:
            dcf_df.to_csv(out_path / f"dcf_valuation_{ts}.csv", encoding="utf-8-sig")

    logger.info("AI增强研报已生成: %s (%d 字)", report_path, len(report_content))
    return report_path


def _llm_enhanced_portfolio(
    pf: pd.DataFrame,
    fundamentals: pd.DataFrame,
    dcf_map: dict | None,
    llm_cfg: Any,
    sector_desc_map: dict[str, str] | None,
    news_ctx: Any | None,
) -> str:
    """调用 LLM 为每只持仓生成深度分析。"""
    from .llm import _call_model_raw, _init_raw_maps
    from .dcf import format_dcf_for_prompt

    sdmap = sector_desc_map or {}
    lines = ["## 二、持仓个股深度分析（AI生成）\n"]

    _format_news = None
    if news_ctx is not None:
        try:
            from .news import format_news_for_prompt
            _format_news = format_news_for_prompt
        except ImportError:
            pass

    models = llm_cfg.available_models
    if not models:
        return _build_portfolio_section(pf, fundamentals, dcf_map, llm_cfg, sector_desc_map, news_ctx)

    import time as _time

    for i, t in enumerate(pf.index, 1):
        row = pf.loc[t]
        name = str(row.get("name", t))
        engine = str(row.get("engine", ""))
        final_score = float(row.get("final_score", 0))
        quant_score = float(row.get("quant_score", 0))
        action = str(row.get("action", ""))
        advice_action = str(row.get("advice_action", ""))

        fund = fundamentals.loc[t] if t in fundamentals.index else pd.Series()
        sector = sdmap.get(t, str(fund.get("sector", "N/A")) if not fund.empty else "N/A")

        market_cap_raw = float(fund.get("market_cap", 0)) if not fund.empty else 0
        market_cap_str = f"{market_cap_raw/1e9:.1f}B" if market_cap_raw > 0 else "N/A"

        vol_raw = row.get("annual_vol", None)
        vol_str = f"{float(vol_raw):.0%}" if vol_raw is not None and pd.notna(vol_raw) else "N/A"

        dcf_block = format_dcf_for_prompt(t, dcf_map) if dcf_map else ""
        advice_block = f"AI投资建议: {advice_action}" if advice_action else ""
        news_text = _format_news(t, news_ctx) if _format_news and news_ctx else ""

        rev_growth = f"{float(fund.get('revenue_growth', 0)):.1%}" if not fund.empty and pd.notna(fund.get("revenue_growth")) else "N/A"
        gross_margin = f"{float(fund.get('gross_margin', 0)):.1%}" if not fund.empty and pd.notna(fund.get("gross_margin")) else "N/A"
        forward_pe = f"{float(fund.get('forward_pe', 0)):.1f}" if not fund.empty and pd.notna(fund.get("forward_pe")) else "N/A"
        roe = f"{float(fund.get('return_on_equity', 0)):.1%}" if not fund.empty and pd.notna(fund.get("return_on_equity")) else "N/A"
        dte = f"{float(fund.get('debt_to_equity', 0)):.0f}%" if not fund.empty and pd.notna(fund.get("debt_to_equity")) else "N/A"

        prompt = _STOCK_ANALYSIS_PROMPT.format(
            ticker=t, name=name, sector=sector, market_cap=market_cap_str,
            quant_score=quant_score, final_score=final_score, action=action,
            dcf_block=dcf_block, advice_block=advice_block, news_block=news_text,
            rev_growth=rev_growth, gross_margin=gross_margin, forward_pe=forward_pe,
            roe=roe, debt_to_equity=dte, annual_vol=vol_str,
        )

        # 用第一个可用模型生成（研报只需一个模型，不必交叉验证）
        model = models[0]
        try:
            raw = _call_model_raw(llm_cfg, model, prompt)
            analysis = _parse_stock_analysis(raw, t, name)
            lines.append(analysis)
            logger.info("研报分析 [%d/%d] %s 完成", i, len(pf), t)
        except Exception as e:
            logger.warning("研报分析 [%d/%d] %s 异常: %s", i, len(pf), t, e)
            # 降级到纯数据版本
            lines.append(_fallback_stock_section(t, name, row, fund, dcf_map))

        _time.sleep(0.3)

    return "\n".join(lines)


def _parse_stock_analysis(raw: str, ticker: str, name: str) -> str:
    """解析 LLM 返回的个股分析 JSON，格式化为 Markdown。"""
    payload = raw.strip()
    if payload.startswith("```"):
        payload = payload.strip("`").replace("json", "", 1).strip()
    if "<think>" in payload:
        idx = payload.rfind("}")
        if idx >= 0:
            start = payload.find("{")
            if start >= 0:
                payload = payload[start : idx + 1]

    try:
        data = json.loads(payload)
        rating = data.get("rating", "持有")
        target_range = data.get("target_range", "N/A")
        analysis = data.get("analysis", "")
        catalysts = data.get("catalysts", "")
        risks = data.get("risks", "")

        rating_emoji = {"买入": "🟢", "持有": "🟡", "减持": "🔴"}.get(rating, "🟡")

        lines = [
            f"### {rating_emoji} {ticker} ({name}) — {rating}",
            f"**目标价区间**: {target_range}\n",
            analysis,
            "",
        ]
        if catalysts:
            lines.append(f"**近期催化剂**: {catalysts}")
        if risks:
            lines.append(f"**关键风险**: {risks}")
        lines.append("")
        return "\n".join(lines)

    except Exception:
        return f"### 🟡 {ticker} ({name})\n{raw[:300]}\n"


def _fallback_stock_section(
    ticker: str, name: str, row: Any, fund: pd.Series, dcf_map: dict | None,
) -> str:
    """LLM 失败时的降级版本。"""
    final_score = float(row.get("final_score", 0))
    action = str(row.get("action", ""))
    advice = str(row.get("advice_action", "N/A"))

    dcf_str = ""
    if dcf_map and ticker in dcf_map:
        r = dcf_map[ticker]
        dcf_str = f" | DCF Base ${r.base_value:.1f} ({r.upside_pct:+.0%})"

    return (
        f"### 🟡 {ticker} ({name})\n"
        f"- 评分: {final_score:.1f} | 信号: {action} | 建议: {advice}{dcf_str}\n"
    )
