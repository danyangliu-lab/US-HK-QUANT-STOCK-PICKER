from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional

# ── 动态股票池覆盖文件（由 universe_builder.py 生成）──
_GROWTH_OVERRIDE_FILE = os.path.expanduser(
    "~/.openclaw/workspace/data/growth_tickers_override.json"
)


def _load_dynamic_growth_tickers() -> Optional[list[str]]:
    """尝试从动态覆盖文件加载成长池。
    如果文件存在且有效，返回 ticker 列表；否则返回 None（回退到硬编码默认值）。
    """
    if not os.path.exists(_GROWTH_OVERRIDE_FILE):
        return None
    try:
        with open(_GROWTH_OVERRIDE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        tickers = data.get("growth_tickers", [])
        if tickers and isinstance(tickers, list) and len(tickers) >= 10:
            return tickers
    except Exception:
        pass
    return None


def _load_dynamic_sector_map() -> dict[str, str]:
    """从动态覆盖文件加载 sector_map 增量。"""
    if not os.path.exists(_GROWTH_OVERRIDE_FILE):
        return {}
    try:
        with open(_GROWTH_OVERRIDE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("sector_map", {})
    except Exception:
        return {}


# 硬编码默认成长池（fallback）
_DEFAULT_GROWTH_TICKERS = [
    # 原成长
    "PLTR", "COHR", "NBIS", "CLS", "LITE", "VRT", "GEV",
    # 高增速大市值（成长逻辑）
    "NVDA", "AMD", "MU",
    # 新增自选
    "SNDK", "CNQ",
    # 新增自选 - 工业/矿业成长
    "ALM", "IPGP",
    # 新增自选 - 半导体封测/油服
    "KLIC", "ASX", "AMKR", "PUMP",
    # 新增自选 - 国防/无人机
    "KTOS", "AVAV",
    # 新增自选 - 资源/国防
    "SCCO", "ALB", "LMT",
    # 大市值（统一成长+趋势逻辑评估）
    "AAPL", "MSFT", "AMZN", "AVGO", "ORCL", "XLE", "B",
    # 新增自选 - 光通信/特种材料/可观测性/铜矿
    "DY", "ATI", "FN", "COPX",
    # 港股
    "0700.HK", "9988.HK", "2706.HK",
]


@dataclass
class StrategyConfig:
    lookback_days: int = 320
    min_price_days: int = 60
    rebalance_frequency: str = "W-FRI"
    monthly_rebalance_day: int = 1

    # 融合权重
    quant_weight: float = 0.8
    llm_weight: float = 0.2

    # 风控（小资金激进模式：~SGD 2000 / ~USD 1500）
    # 核心策略: 极度集中(≤3只) + 高弹性 + 杠杆ETF为主 + 日内短线
    max_weight_growth: float = 0.60         # 单票上限60%（集中火力，3只平分即33%）
    max_weight_leverage: float = 0.60       # 杠杆ETF也允许重仓
    max_weight_smallcap: float = 0.40       # 小盘投机加大仓位（小资金博高赔率）
    max_smallcap_total: float = 0.50        # 小盘投机池总仓位放开
    max_industry_weight: float = 0.80       # 行业集中度放开（集中在最强赛道）
    max_leverage_etf_total: float = 0.80    # 可以全仓杠杆ETF（小资金博弹性）

    # 选股阈值（极度集中: 只选Top 3）
    top_n_buy: int = 3                      # 只买评分最高的3只
    min_score_to_buy: float = 60            # 买入门槛提高到60（只买最强）
    min_score_to_hold: float = 50           # 持仓门槛也提高

    # ── 自选观察池 ──
    # 成长池：优先从 dynamic_universe 动态加载，否则回退到硬编码默认值
    # 动态池由 universe_builder.py 每周自动生成（弹性优先选股）
    growth_tickers: list[str] = field(
        default_factory=lambda: _load_dynamic_growth_tickers() or list(_DEFAULT_GROWTH_TICKERS)
    )
    # 杠杆ETF池（纯趋势+波动率引擎）
    leverage_etf_tickers: list[str] = field(default_factory=lambda: [
        "SOXL", "DFEN", "LITX", "SNXX", "7747.HK", "7709.HK",
        # 新增自选 - 国防/无人机/通信杠杆
        "ONDL", "KTUP", "AVXX", "LMTL",
    ])

    # ── 小盘投机池（市值<10亿美金，小仓位高赔率）──
    smallcap_tickers: list[str] = field(default_factory=lambda: [
        "WOLF", "NVTS", "NEOV", "AAOI", "CLPT",
        "ONDS", "AUR", "CRCL",
    ])

    # ── 持仓股票池（新账户，初始为空，开仓后会自动更新）──
    portfolio_stock_tickers: list[str] = field(default_factory=lambda: [])
    portfolio_leverage_tickers: list[str] = field(default_factory=lambda: [])

    # 杠杆ETF → 底层股票映射（有底层的会引入底层基本面+技术面）
    leverage_underlying_map: dict[str, str] = field(default_factory=lambda: {
        "LITX": "LITE",
        "SNXX": "SNDK",
        "ONDL": "ONDS",
        "KTUP": "KTOS",
        "AVXX": "AVAV",
        "LMTL": "LMT",
    })

    # 指数型杠杆ETF → 跟踪指数/参考ETF映射（用于引入指数技术面）
    leverage_index_map: dict[str, str] = field(default_factory=lambda: {
        "SOXL": "SMH",       # 3x半导体 → VanEck半导体ETF
        "DFEN": "ITA",       # 3x航空国防 → iShares航空国防ETF
        "7747.HK": "2800.HK",  # 南方两倍做多纳指 → 盈富基金（港股大盘参考）
        "7709.HK": "2800.HK",  # 南方两倍做多 → 盈富基金
    })

    # 杠杆ETF行业描述（传给LLM用于时事分析）
    leverage_sector_desc: dict[str, str] = field(default_factory=lambda: {
        "SOXL": "3倍做多半导体指数ETF，跟踪费城半导体指数(SOX)，受AI芯片需求、半导体周期、中美科技竞争、出口管制等影响",
        "DFEN": "3倍做多航空国防ETF，跟踪道琼斯航空国防指数，受地缘政治冲突(中东/俄乌)、美国国防预算、军工订单、战争风险等影响",
        "LITX": "2倍做多Lumentum(LITE)，光通信和3D传感龙头，受AI数据中心光模块需求驱动",
        "SNXX": "2倍做多闪迪(SNDK)，存储芯片企业，受NAND闪存价格周期和AI存储需求影响",
        "ONDL": "2倍做多Ondas(ONDS)，工业物联网和无人机通信企业，受美国关键基础设施和国防通信需求驱动",
        "KTUP": "2倍做多克瑞拓斯(KTOS)，无人系统和卫星通信国防企业，受无人机/太空军/国防预算驱动",
        "AVXX": "2倍做多AeroVironment(AVAV)，小型无人机和巡飞弹龙头，受俄乌/中东战争无人机需求驱动",
        "LMTL": "2倍做多洛克希德马丁(LMT)，全球最大国防承包商，受F-35/导弹防御/地缘冲突驱动",
        "7747.HK": "南方两倍做多纳指ETF，港股杠杆产品，跟踪纳斯达克100指数",
        "7709.HK": "南方两倍做多恒指ETF，港股杠杆产品，跟踪恒生指数",
    })

    sector_map: dict[str, str] = field(default_factory=lambda: {
        # 成长
        "NVDA": "Tech", "AMD": "Tech", "MU": "Tech",
        "PLTR": "Tech", "AAOI": "Tech", "WOLF": "Tech", "COHR": "Tech",
        "NBIS": "Tech", "CLS": "Tech", "LITE": "Tech",
        "VRT": "Industrials", "GEV": "Industrials", "NEOV": "Energy",
        "CRCL": "Fintech", "CLPT": "Healthcare", "NVTS": "Tech",
        "SNDK": "Tech", "CNQ": "Energy",
        "ALM": "Materials", "IPGP": "Industrials",
        "AUR": "Auto", "KLIC": "Tech", "ASX": "Tech", "AMKR": "Tech", "PUMP": "Energy",
        "KTOS": "Defense", "AVAV": "Defense", "ONDS": "Defense",
        "SCCO": "Materials", "ALB": "Materials", "LMT": "Defense",
        "DY": "Tech", "ATI": "Materials", "FN": "Tech", "COPX": "Materials",
        # 大市值
        "AAPL": "Tech", "MSFT": "Tech", "AVGO": "Tech", "ORCL": "Tech",
        "AMZN": "Consumer", "XLE": "Energy", "B": "Materials",
        # 港股
        "0700.HK": "Tech", "9988.HK": "Tech",
        "2706.HK": "Tech",
        # 杠杆ETF
        "SOXL": "Leveraged", "DFEN": "Leveraged",
        "LITX": "Leveraged", "SNXX": "Leveraged",
        "ONDL": "Leveraged", "KTUP": "Leveraged", "AVXX": "Leveraged", "LMTL": "Leveraged",
        "7747.HK": "Leveraged", "7709.HK": "Leveraged",
        # 动态加载的行业映射（来自 universe_builder.py）
        **_load_dynamic_sector_map(),
    })

    @property
    def portfolio_tickers(self) -> list[str]:
        return list(dict.fromkeys(self.portfolio_stock_tickers + self.portfolio_leverage_tickers))

    @property
    def watchlist_growth_tickers(self) -> list[str]:
        """成长自选池 + 持仓个股，去重保序（确保所有持仓股都在自选评分范围内）"""
        return list(dict.fromkeys(self.growth_tickers + self.portfolio_stock_tickers))

    @property
    def watchlist_leverage_tickers(self) -> list[str]:
        """杠杆自选池 + 持仓杠杆ETF，去重保序（确保所有持仓杠杆ETF都在自选评分范围内）"""
        return list(dict.fromkeys(self.leverage_etf_tickers + self.portfolio_leverage_tickers))

    @property
    def all_tickers(self) -> list[str]:
        ordered = (
            self.growth_tickers
            + self.smallcap_tickers
            + self.leverage_etf_tickers
            + self.portfolio_stock_tickers
            + self.portfolio_leverage_tickers
        )
        # 加入参考指数/ETF，用于杠杆引擎技术面分析
        ref_tickers = list(self.leverage_index_map.values())
        ordered = ordered + ref_tickers
        return list(dict.fromkeys(ordered))


def default_config() -> StrategyConfig:
    return StrategyConfig()
