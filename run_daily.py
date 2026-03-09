from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from quant_system.config import default_config
from quant_system.engine import run_daily_pipeline
from quant_system.llm import LLMConfig


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )


def _load_dotenv() -> None:
    """自动加载项目根目录 .env 文件（无需手动 export）。"""
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv
        load_dotenv(env_path, override=False)
    except ImportError:
        import os
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())


def main() -> None:
    _setup_logging()
    _load_dotenv()

    parser = argparse.ArgumentParser(description="成长+趋势 量化选股系统（日更版 v3 — 三模型交叉验证）")
    parser.add_argument("--out", default="outputs", help="输出目录")
    parser.add_argument("--llm", action="store_true", help="启用LLM辅助评分（Kimi K2.5 + DeepSeek V3.2 + GLM-5）")
    parser.add_argument("--no-kimi", action="store_true", help="禁用Kimi K2.5")
    parser.add_argument("--no-deepseek", action="store_true", help="禁用DeepSeek V3.2")
    parser.add_argument("--no-glm", action="store_true", help="禁用GLM-5")
    parser.add_argument("--cross-mode", choices=["cross", "avg", "primary"], default=None,
                        help="交叉验证模式: cross=方向融合(默认), avg=简单平均, primary=仅第一个可用模型")
    parser.add_argument("--sync-moomoo", action="store_true",
                        help="从 moomoo (富途) 同步持仓和自选股（需本地运行 OpenD）")
    parser.add_argument("--sync-moomoo-portfolio-only", action="store_true",
                        help="仅同步 moomoo 持仓（不同步自选股）")
    parser.add_argument("--sync-longbridge", action="store_true",
                        help="从长桥 LongPort 同步持仓和自选股（云端API，无需本地网关）")
    parser.add_argument("--sync-longbridge-portfolio-only", action="store_true",
                        help="仅同步长桥持仓（不同步自选股）")
    parser.add_argument("--report", action="store_true",
                        help="生成格式化研报（Markdown + DCF 估值汇总 CSV）")
    args = parser.parse_args()

    cfg = default_config()

    # moomoo 同步（旧方案，保留向后兼容）
    if args.sync_moomoo or args.sync_moomoo_portfolio_only:
        from quant_system.moomoo_sync import MoomooConfig, sync_from_moomoo
        moomoo_cfg = MoomooConfig.from_env()
        if args.sync_moomoo_portfolio_only:
            moomoo_cfg.sync_watchlist = False
        cfg = sync_from_moomoo(cfg, moomoo_cfg)

    # 长桥同步（新方案，推荐）
    if args.sync_longbridge or args.sync_longbridge_portfolio_only:
        from quant_system.longbridge_sync import LongbridgeConfig, sync_from_longbridge
        lb_cfg = LongbridgeConfig.from_env()
        if args.sync_longbridge_portfolio_only:
            lb_cfg.sync_portfolio = True
            lb_cfg.watchlist_json_path = ""
        cfg = sync_from_longbridge(cfg, lb_cfg)
    llm_cfg = LLMConfig.from_env()
    if args.llm:
        llm_cfg.enabled = True
    if args.no_kimi:
        llm_cfg.kimi_enabled = False
    if args.no_deepseek:
        llm_cfg.deepseek_enabled = False
    if args.no_glm:
        llm_cfg.glm_enabled = False
    if args.cross_mode:
        llm_cfg.cross_validation_mode = args.cross_mode

    result = run_daily_pipeline(cfg=cfg, llm_cfg=llm_cfg, out_dir=args.out,
                                enable_report=args.report)

    signal = result.signal_table
    print("\n" + "=" * 90)
    print("  📊 自选观察池 — AI推荐组合（按 final_score 从高到低）")
    print("=" * 90)
    cols = ["name", "engine", "quant_score", "event_score", "analyst_rating", "final_score", "action"]
    cols = [c for c in cols if c in signal.columns]
    with pd.option_context("display.max_rows", None):
        print(signal[cols].to_string())

    print("\n" + "=" * 90)
    print("  📊 AI推荐 — 目标权重")
    print("=" * 90)
    if result.weights.empty:
        print("  无可分配权重")
    else:
        print((result.weights.head(15) * 100).round(2).astype(str) + "%")

    # 变动摘要
    changed = result.diff_report[result.diff_report["signal_change"] != "UNCHANGED"]
    if not changed.empty:
        print("\n" + "=" * 90)
        print("  📊 AI推荐 — 信号变动（vs 昨日）")
        print("=" * 90)
        print(changed[["engine", "final_score", "signal_change"]].to_string())

    # 组合收益跟踪
    if result.tracking is not None:
        tk = result.tracking
        print("\n" + "=" * 90)
        print("  📊 AI推荐 — 组合收益跟踪")
        print("=" * 90)
        print(f"  本期收益: {tk.period_return * 100:+.2f}%")
        print(f"  累计收益: {tk.cumulative_return * 100:+.2f}%")
        # 基准指数对比
        if tk.benchmarks:
            print("\n  基准指数对比:")
            for bm in tk.benchmarks:
                excess = tk.period_return - bm.period_return
                cum_excess = tk.cumulative_return - bm.cumulative_return
                print(f"    {bm.name}({bm.ticker}): 本期 {bm.period_return * 100:+.2f}%  "
                      f"累计 {bm.cumulative_return * 100:+.2f}%  "
                      f"| 超额: 本期 {excess * 100:+.2f}%  累计 {cum_excess * 100:+.2f}%")
        if not tk.holding_details.empty:
            print("\n  持仓明细:")
            detail = tk.holding_details.copy()
            detail["return_pct"] = detail["return_pct"].map(lambda x: f"{x * 100:+.2f}%")
            detail["weighted_return"] = detail["weighted_return"].map(lambda x: f"{x * 100:+.2f}%")
            detail["weight"] = detail["weight"].map(lambda x: f"{x * 100:.1f}%")
            detail["entry_price"] = detail["entry_price"].round(2)
            detail["current_price"] = detail["current_price"].round(2)
            print(detail.to_string(index=False))

    # 持仓投资建议
    if result.portfolio_report is not None and not result.portfolio_report.empty:
        pf = result.portfolio_report
        print("\n" + "=" * 90)
        print("  💼 我的持仓 — 评分与投资建议")
        print("=" * 90)
        pf_cols = ["name", "engine", "quant_score", "final_score", "action",
                   "advice_action", "advice_confidence", "advice_reason"]
        pf_cols = [c for c in pf_cols if c in pf.columns]
        with pd.option_context("display.max_rows", None, "display.max_colwidth", 60):
            print(pf[pf_cols].to_string())

        # 持仓整体分析
        if result.portfolio_overall:
            print("\n" + "-" * 90)
            print("  💼 我的持仓 — 整体组合分析")
            print("-" * 90)
            print(result.portfolio_overall)

    # DCF 估值汇总
    if result.dcf_map:
        from quant_system.dcf import dcf_summary_df
        dcf_df = dcf_summary_df(result.dcf_map)
        if not dcf_df.empty:
            print("\n" + "=" * 90)
            print("  💰 DCF 估值汇总")
            print("=" * 90)
            with pd.option_context("display.max_rows", None, "display.float_format", "{:.1f}".format):
                print(dcf_df.to_string())

    # 研报路径
    if result.report_path:
        print("\n" + "=" * 90)
        print(f"  📝 研报已生成: {result.report_path}")
        print("=" * 90)

    print(f"\n输出目录: {result.output_dir.resolve()}")


if __name__ == "__main__":
    pd.set_option("display.width", 180)
    pd.set_option("display.max_columns", 20)
    main()
