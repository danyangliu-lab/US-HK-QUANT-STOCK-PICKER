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

    parser = argparse.ArgumentParser(description="成长+趋势 量化选股系统（日更版 v2）")
    parser.add_argument("--out", default="outputs", help="输出目录")
    parser.add_argument("--llm", action="store_true", help="启用混元LLM辅助评分")
    args = parser.parse_args()

    cfg = default_config()
    llm_cfg = LLMConfig.from_env()
    if args.llm:
        llm_cfg.enabled = True

    result = run_daily_pipeline(cfg=cfg, llm_cfg=llm_cfg, out_dir=args.out)

    signal = result.signal_table
    print("\n" + "=" * 90)
    print("  全部股票评分（按 final_score 从高到低）")
    print("=" * 90)
    cols = ["name", "engine", "quant_score", "event_score", "final_score", "action"]
    cols = [c for c in cols if c in signal.columns]
    with pd.option_context("display.max_rows", None):
        print(signal[cols].to_string())

    print("\n" + "=" * 90)
    print("  目标权重")
    print("=" * 90)
    if result.weights.empty:
        print("  无可分配权重")
    else:
        print((result.weights.head(15) * 100).round(2).astype(str) + "%")

    # 变动摘要
    changed = result.diff_report[result.diff_report["signal_change"] != "UNCHANGED"]
    if not changed.empty:
        print("\n" + "=" * 90)
        print("  信号变动（vs 昨日）")
        print("=" * 90)
        print(changed[["engine", "final_score", "signal_change"]].to_string())

    # 组合收益跟踪
    if result.tracking is not None:
        tk = result.tracking
        print("\n" + "=" * 90)
        print("  组合收益跟踪")
        print("=" * 90)
        print(f"  本期收益: {tk.period_return * 100:+.2f}%")
        print(f"  累计收益: {tk.cumulative_return * 100:+.2f}%")
        if not tk.holding_details.empty:
            print("\n  持仓明细:")
            detail = tk.holding_details.copy()
            detail["return_pct"] = detail["return_pct"].map(lambda x: f"{x * 100:+.2f}%")
            detail["weighted_return"] = detail["weighted_return"].map(lambda x: f"{x * 100:+.2f}%")
            detail["weight"] = detail["weight"].map(lambda x: f"{x * 100:.1f}%")
            detail["entry_price"] = detail["entry_price"].round(2)
            detail["current_price"] = detail["current_price"].round(2)
            print(detail.to_string(index=False))

    print(f"\n输出目录: {result.output_dir.resolve()}")


if __name__ == "__main__":
    pd.set_option("display.width", 180)
    pd.set_option("display.max_columns", 20)
    main()
