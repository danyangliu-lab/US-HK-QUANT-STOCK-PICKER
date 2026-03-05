"""moomoo (富途) OpenAPI 同步模块。

通过 futu-api SDK 连接本地 OpenD 网关，自动获取：
- 实际持仓（美股 + 港股）→ 覆写 config 的 portfolio_stock_tickers / portfolio_leverage_tickers
- 自选股分组 → 覆写 config 的 growth_tickers / leverage_etf_tickers / smallcap_tickers

前置条件：
1. pip install futu-api
2. 本地或云端运行 FutuOpenD 网关（默认 127.0.0.1:11111）
3. moomoo 账户已登录 OpenD
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# moomoo 代码前缀 → 本系统格式的映射
# moomoo 格式: "US.NVDA", "HK.00700"
# 本系统格式: "NVDA", "0700.HK"
_MARKET_PREFIX = {
    "US": "",       # 美股直接取代码
    "HK": ".HK",   # 港股加 .HK 后缀
}


@dataclass
class MoomooConfig:
    """moomoo 连接配置。"""
    host: str = "127.0.0.1"
    port: int = 11111
    # 交易市场过滤
    trd_market_us: bool = True    # 同步美股持仓
    trd_market_hk: bool = True    # 同步港股持仓
    # 自选股分组名称 → 映射到本系统的哪个池子
    watchlist_growth: str = "美股"        # 自选股分组名 → growth_tickers
    watchlist_hk: str = "港股"            # 港股自选 → 也加入 growth_tickers
    watchlist_leverage: str = ""          # 如有杠杆ETF分组 → leverage_etf_tickers（留空=不同步）
    watchlist_smallcap: str = ""          # 小盘投机分组 → smallcap_tickers（留空=不同步）
    # 是否同步自选股（false=只同步持仓）
    sync_watchlist: bool = True
    # 是否同步持仓
    sync_portfolio: bool = True

    @classmethod
    def from_env(cls) -> MoomooConfig:
        return cls(
            host=os.environ.get("MOOMOO_HOST", "127.0.0.1"),
            port=int(os.environ.get("MOOMOO_PORT", "11111")),
            trd_market_us=os.environ.get("MOOMOO_TRD_US", "true").lower() == "true",
            trd_market_hk=os.environ.get("MOOMOO_TRD_HK", "true").lower() == "true",
            watchlist_growth=os.environ.get("MOOMOO_WATCHLIST_GROWTH", "美股"),
            watchlist_hk=os.environ.get("MOOMOO_WATCHLIST_HK", "港股"),
            watchlist_leverage=os.environ.get("MOOMOO_WATCHLIST_LEVERAGE", ""),
            watchlist_smallcap=os.environ.get("MOOMOO_WATCHLIST_SMALLCAP", ""),
            sync_watchlist=os.environ.get("MOOMOO_SYNC_WATCHLIST", "true").lower() == "true",
            sync_portfolio=os.environ.get("MOOMOO_SYNC_PORTFOLIO", "true").lower() == "true",
        )


def _convert_code(moomoo_code: str) -> str | None:
    """将 moomoo 代码 (如 'US.NVDA', 'HK.00700') 转为本系统格式 ('NVDA', '0700.HK')。

    返回 None 表示不支持的市场。
    """
    parts = moomoo_code.split(".", 1)
    if len(parts) != 2:
        return None
    market, symbol = parts[0], parts[1]

    if market == "US":
        return symbol
    elif market == "HK":
        # moomoo 港股代码: "00700" → 本系统: "0700.HK"
        # 去掉多余前导零：5位→4位（moomoo用5位，yfinance用4位）
        if len(symbol) == 5 and symbol.startswith("0"):
            symbol = symbol[1:]
        return f"{symbol}.HK"
    else:
        logger.debug("跳过不支持的市场: %s", moomoo_code)
        return None


def _fetch_positions(moomoo_cfg: MoomooConfig) -> list[str]:
    """从 moomoo 获取实际持仓代码列表。"""
    try:
        from futu import OpenSecTradeContext, TrdMarket, RET_OK, SecurityFirm
    except ImportError:
        logger.error("futu-api 未安装，请运行: pip install futu-api")
        return []

    tickers: list[str] = []

    markets = []
    if moomoo_cfg.trd_market_us:
        markets.append(("美股", TrdMarket.US))
    if moomoo_cfg.trd_market_hk:
        markets.append(("港股", TrdMarket.HK))

    for market_name, trd_market in markets:
        try:
            trd_ctx = OpenSecTradeContext(
                filter_trdmarket=trd_market,
                host=moomoo_cfg.host,
                port=moomoo_cfg.port,
                security_firm=SecurityFirm.FUTUSECURITIES,
            )
            ret, data = trd_ctx.position_list_query()
            trd_ctx.close()

            if ret != RET_OK:
                logger.warning("moomoo %s持仓查询失败: %s", market_name, data)
                continue

            if data.empty:
                logger.info("moomoo %s持仓为空", market_name)
                continue

            for _, row in data.iterrows():
                code = _convert_code(str(row["code"]))
                qty = float(row.get("qty", 0))
                if code and qty > 0:
                    tickers.append(code)

            logger.info("moomoo %s持仓: %d 只 → %s", market_name, len(tickers), tickers)

        except Exception as e:
            logger.warning("moomoo %s持仓查询异常: %s", market_name, e)

    return tickers


def _fetch_watchlist(moomoo_cfg: MoomooConfig, group_name: str) -> list[str]:
    """从 moomoo 获取指定自选股分组的代码列表。"""
    if not group_name:
        return []

    try:
        from futu import OpenQuoteContext, RET_OK
    except ImportError:
        logger.error("futu-api 未安装，请运行: pip install futu-api")
        return []

    tickers: list[str] = []

    try:
        quote_ctx = OpenQuoteContext(host=moomoo_cfg.host, port=moomoo_cfg.port)
        ret, data = quote_ctx.get_user_security(group_name)
        quote_ctx.close()

        if ret != RET_OK:
            logger.warning("moomoo 自选股分组 '%s' 查询失败: %s", group_name, data)
            return []

        if data.empty:
            logger.info("moomoo 自选股分组 '%s' 为空", group_name)
            return []

        for _, row in data.iterrows():
            code = _convert_code(str(row["code"]))
            if code:
                tickers.append(code)

        logger.info("moomoo 自选股 '%s': %d 只", group_name, len(tickers))

    except Exception as e:
        logger.warning("moomoo 自选股 '%s' 查询异常: %s", group_name, e)

    return tickers


def sync_from_moomoo(
    cfg: "StrategyConfig",  # type: ignore[name-defined]
    moomoo_cfg: MoomooConfig | None = None,
) -> "StrategyConfig":  # type: ignore[name-defined]
    """从 moomoo 同步持仓和自选股到 StrategyConfig。

    同步逻辑：
    - 持仓：moomoo 持仓中属于 leverage_etf_tickers 的 → portfolio_leverage_tickers
            其余 → portfolio_stock_tickers
    - 自选股：按分组映射到对应池子

    不在 sector_map / leverage_underlying_map 等静态映射中的新股票
    会被自动加入对应池子，但不会自动添加映射（需手动配置行业等）。

    返回更新后的 cfg（原地修改）。
    """
    if moomoo_cfg is None:
        moomoo_cfg = MoomooConfig.from_env()

    logger.info("========== 开始 moomoo 同步 ==========")
    logger.info("OpenD 地址: %s:%d", moomoo_cfg.host, moomoo_cfg.port)

    # ── 同步持仓 ──
    if moomoo_cfg.sync_portfolio:
        positions = _fetch_positions(moomoo_cfg)
        if positions:
            # 区分杠杆ETF vs 个股
            leverage_set = set(cfg.leverage_etf_tickers)
            pf_stocks = []
            pf_leverage = []
            for t in positions:
                if t in leverage_set:
                    pf_leverage.append(t)
                else:
                    pf_stocks.append(t)

            # 覆写持仓池
            cfg.portfolio_stock_tickers = list(dict.fromkeys(pf_stocks))
            cfg.portfolio_leverage_tickers = list(dict.fromkeys(pf_leverage))

            logger.info("持仓同步完成: 个股 %d 只 %s, 杠杆ETF %d 只 %s",
                        len(pf_stocks), pf_stocks,
                        len(pf_leverage), pf_leverage)

            # 确保持仓中的个股也在自选观察池中（否则不会被下载行情）
            growth_set = set(cfg.growth_tickers)
            new_to_growth = [t for t in pf_stocks if t not in growth_set and t not in leverage_set]
            if new_to_growth:
                cfg.growth_tickers = cfg.growth_tickers + new_to_growth
                logger.info("自动将 %d 只新持仓个股加入自选观察池: %s", len(new_to_growth), new_to_growth)

            # 确保持仓中的杠杆ETF也在杠杆ETF池中
            lev_set = set(cfg.leverage_etf_tickers)
            new_to_lev = [t for t in pf_leverage if t not in lev_set]
            if new_to_lev:
                cfg.leverage_etf_tickers = cfg.leverage_etf_tickers + new_to_lev
                logger.info("自动将 %d 只新持仓杠杆ETF加入杠杆池: %s", len(new_to_lev), new_to_lev)
        else:
            logger.info("moomoo 持仓为空，保持 config 默认值")

    # ── 同步自选股 ──
    if moomoo_cfg.sync_watchlist:
        # 美股自选 → growth_tickers
        if moomoo_cfg.watchlist_growth:
            us_watchlist = _fetch_watchlist(moomoo_cfg, moomoo_cfg.watchlist_growth)
            if us_watchlist:
                # 区分杠杆ETF和普通个股
                leverage_set = set(cfg.leverage_etf_tickers)
                new_growth = [t for t in us_watchlist if t not in leverage_set]
                new_leverage = [t for t in us_watchlist if t in leverage_set]
                cfg.growth_tickers = list(dict.fromkeys(new_growth))
                if new_leverage:
                    cfg.leverage_etf_tickers = list(dict.fromkeys(cfg.leverage_etf_tickers + new_leverage))
                logger.info("自选股 '%s' → growth_tickers: %d 只", moomoo_cfg.watchlist_growth, len(new_growth))

        # 港股自选 → 也加入 growth_tickers
        if moomoo_cfg.watchlist_hk:
            hk_watchlist = _fetch_watchlist(moomoo_cfg, moomoo_cfg.watchlist_hk)
            if hk_watchlist:
                existing = set(cfg.growth_tickers)
                new_hk = [t for t in hk_watchlist if t not in existing]
                if new_hk:
                    cfg.growth_tickers = cfg.growth_tickers + new_hk
                    logger.info("自选股 '%s' → growth_tickers: 新增 %d 只港股", moomoo_cfg.watchlist_hk, len(new_hk))

        # 杠杆ETF自选分组
        if moomoo_cfg.watchlist_leverage:
            lev_watchlist = _fetch_watchlist(moomoo_cfg, moomoo_cfg.watchlist_leverage)
            if lev_watchlist:
                cfg.leverage_etf_tickers = list(dict.fromkeys(lev_watchlist))
                logger.info("自选股 '%s' → leverage_etf_tickers: %d 只", moomoo_cfg.watchlist_leverage, len(lev_watchlist))

        # 小盘投机自选分组
        if moomoo_cfg.watchlist_smallcap:
            sc_watchlist = _fetch_watchlist(moomoo_cfg, moomoo_cfg.watchlist_smallcap)
            if sc_watchlist:
                cfg.smallcap_tickers = list(dict.fromkeys(sc_watchlist))
                logger.info("自选股 '%s' → smallcap_tickers: %d 只", moomoo_cfg.watchlist_smallcap, len(sc_watchlist))

    logger.info("========== moomoo 同步完成 ==========")
    return cfg
