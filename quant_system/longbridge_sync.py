"""LongPort (长桥) OpenAPI 同步模块。

通过 longport SDK 连接长桥云端 API，自动获取：
- 实际持仓（美股 + 港股）→ 覆写 config 的 portfolio_stock_tickers / portfolio_leverage_tickers
- 自选股（长桥自选分组或 longbridge_watchlist.json）→ 覆写 config 的 growth_tickers 等

前置条件：
1. pip install longport
2. 配置环境变量：
   LONGPORT_APP_KEY / LONGPORT_APP_SECRET / LONGPORT_ACCESS_TOKEN / LONGPORT_REGION

替代原 moomoo_sync.py（futu-api + OpenD 本地网关方案）。
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


# ── 代码格式转换 ────────────────────────────────────────

def longport_to_yf(lb_code: str) -> str | None:
    """长桥代码 → Yahoo Finance 代码。

    长桥格式: "700.HK", "AAPL.US"
    YF 格式:  "0700.HK", "AAPL"
    """
    if "." not in lb_code:
        return lb_code
    parts = lb_code.rsplit(".", 1)
    if len(parts) != 2:
        return None
    symbol, region = parts
    if region == "US":
        return symbol
    elif region == "HK":
        # 长桥港股代码可能是 3~4 位，YF 需要 4 位
        if len(symbol) < 4:
            symbol = symbol.zfill(4)
        return f"{symbol}.HK"
    else:
        logger.debug("跳过不支持的市场: %s", lb_code)
        return None


def yf_to_longport(yf_code: str) -> str:
    """Yahoo Finance 代码 → 长桥代码。

    YF 格式:  "0700.HK", "AAPL"
    长桥格式: "700.HK", "AAPL.US"
    """
    if yf_code.endswith(".HK"):
        # 港股: "0700.HK" → "700.HK"（去掉前导零）
        symbol = yf_code.replace(".HK", "").lstrip("0") or "0"
        return f"{symbol}.HK"
    else:
        # 美股: "AAPL" → "AAPL.US"
        return f"{yf_code}.US"


@dataclass
class LongbridgeConfig:
    """长桥同步配置。"""
    # 交易市场过滤
    trd_market_us: bool = True
    trd_market_hk: bool = True
    # 是否同步持仓
    sync_portfolio: bool = True
    # 外部自选文件路径（OpenClaw 的 longbridge_watchlist.json）
    watchlist_json_path: str = ""

    @classmethod
    def from_env(cls) -> LongbridgeConfig:
        return cls(
            trd_market_us=os.environ.get("LONGPORT_TRD_US", "true").lower() == "true",
            trd_market_hk=os.environ.get("LONGPORT_TRD_HK", "true").lower() == "true",
            sync_portfolio=os.environ.get("LONGPORT_SYNC_PORTFOLIO", "true").lower() == "true",
            watchlist_json_path=os.environ.get(
                "LONGPORT_WATCHLIST_JSON",
                os.path.expanduser("~/.openclaw/workspace/data/longbridge_watchlist.json"),
            ),
        )


def _load_env_from_file() -> None:
    """从 ~/.openclaw/workspace/.env 加载 LONGPORT_ 环境变量。"""
    env_file = os.path.expanduser("~/.openclaw/workspace/.env")
    if not os.path.exists(env_file):
        return
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key.startswith("LONGPORT_") and key not in os.environ:
                    os.environ[key] = value


def _fetch_positions(lb_cfg: LongbridgeConfig) -> list[str]:
    """从长桥获取实际持仓代码列表（返回 YF 格式）。"""
    _load_env_from_file()
    try:
        from longport.openapi import Config, TradeContext
    except ImportError:
        logger.error("longport 未安装，请运行: pip install longport")
        return []

    try:
        config = Config.from_env()
        ctx = TradeContext(config)
        positions = ctx.stock_positions()
    except Exception as e:
        logger.warning("长桥持仓查询失败: %s", e)
        return []

    tickers: list[str] = []
    channels = positions.channels if hasattr(positions, "channels") else []
    for channel in channels:
        pos_list = channel.positions if hasattr(channel, "positions") else []
        for p in pos_list:
            lb_code = p.symbol if hasattr(p, "symbol") else ""
            qty = int(p.quantity) if hasattr(p, "quantity") else 0
            if not lb_code or qty <= 0:
                continue
            # 市场过滤
            if lb_code.endswith(".US") and not lb_cfg.trd_market_us:
                continue
            if lb_code.endswith(".HK") and not lb_cfg.trd_market_hk:
                continue
            yf_code = longport_to_yf(lb_code)
            if yf_code:
                tickers.append(yf_code)

    logger.info("长桥持仓: %d 只 → %s", len(tickers), tickers)
    return tickers


def _load_watchlist_json(json_path: str) -> dict[str, list[str]]:
    """从 longbridge_watchlist.json 加载自选股（转为 YF 格式）。

    返回 {"us": [...], "hk": [...]}，值为 YF 格式代码列表。
    """
    if not json_path or not os.path.exists(json_path):
        return {}
    try:
        with open(json_path) as f:
            data = json.load(f)
    except Exception as e:
        logger.warning("加载 %s 失败: %s", json_path, e)
        return {}

    result: dict[str, list[str]] = {}
    for market in ("us", "hk"):
        items = data.get(market, [])
        codes = []
        for item in items:
            lb_code = item.get("code", "") if isinstance(item, dict) else str(item)
            yf_code = longport_to_yf(lb_code)
            if yf_code:
                codes.append(yf_code)
        if codes:
            result[market] = codes
    return result


def sync_from_longbridge(
    cfg: "StrategyConfig",  # type: ignore[name-defined]
    lb_cfg: LongbridgeConfig | None = None,
) -> "StrategyConfig":  # type: ignore[name-defined]
    """从长桥同步持仓和自选股到 StrategyConfig。

    同步逻辑：
    - 持仓：长桥持仓中属于 leverage_etf_tickers 的 → portfolio_leverage_tickers
            其余 → portfolio_stock_tickers
    - 自选股：从 longbridge_watchlist.json 加载 → 加入 growth_tickers

    返回更新后的 cfg（原地修改）。
    """
    if lb_cfg is None:
        lb_cfg = LongbridgeConfig.from_env()

    logger.info("========== 开始长桥同步 ==========")

    # ── 同步持仓 ──
    if lb_cfg.sync_portfolio:
        positions = _fetch_positions(lb_cfg)
        if positions:
            leverage_set = set(cfg.leverage_etf_tickers)
            pf_stocks = []
            pf_leverage = []
            for t in positions:
                if t in leverage_set:
                    pf_leverage.append(t)
                else:
                    pf_stocks.append(t)

            cfg.portfolio_stock_tickers = list(dict.fromkeys(pf_stocks))
            cfg.portfolio_leverage_tickers = list(dict.fromkeys(pf_leverage))

            logger.info(
                "持仓同步完成: 个股 %d 只 %s, 杠杆ETF %d 只 %s",
                len(pf_stocks), pf_stocks,
                len(pf_leverage), pf_leverage,
            )

            # 确保持仓中的个股也在自选观察池中
            growth_set = set(cfg.growth_tickers)
            new_to_growth = [t for t in pf_stocks if t not in growth_set and t not in leverage_set]
            if new_to_growth:
                cfg.growth_tickers = cfg.growth_tickers + new_to_growth
                logger.info("自动将 %d 只新持仓个股加入自选观察池: %s", len(new_to_growth), new_to_growth)

            lev_set = set(cfg.leverage_etf_tickers)
            new_to_lev = [t for t in pf_leverage if t not in lev_set]
            if new_to_lev:
                cfg.leverage_etf_tickers = cfg.leverage_etf_tickers + new_to_lev
                logger.info("自动将 %d 只新持仓杠杆ETF加入杠杆池: %s", len(new_to_lev), new_to_lev)
        else:
            logger.info("长桥持仓为空，保持 config 默认值")

    # ── 从 watchlist JSON 同步自选股 ──
    if lb_cfg.watchlist_json_path:
        wl = _load_watchlist_json(lb_cfg.watchlist_json_path)
        leverage_set = set(cfg.leverage_etf_tickers)
        existing_growth = set(cfg.growth_tickers)

        for market, codes in wl.items():
            new_growth = [c for c in codes if c not in existing_growth and c not in leverage_set]
            new_lev = [c for c in codes if c in leverage_set and c not in set(cfg.leverage_etf_tickers)]
            if new_growth:
                cfg.growth_tickers = cfg.growth_tickers + new_growth
                existing_growth.update(new_growth)
                logger.info("自选JSON (%s) → growth_tickers: 新增 %d 只", market, len(new_growth))
            if new_lev:
                cfg.leverage_etf_tickers = cfg.leverage_etf_tickers + new_lev
                logger.info("自选JSON (%s) → leverage_etf_tickers: 新增 %d 只", market, len(new_lev))

    logger.info("========== 长桥同步完成 ==========")
    return cfg
