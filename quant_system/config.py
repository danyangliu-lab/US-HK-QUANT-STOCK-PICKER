from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class StrategyConfig:
    lookback_days: int = 320
    min_price_days: int = 60
    rebalance_frequency: str = "W-FRI"
    monthly_rebalance_day: int = 1

    # 融合权重
    quant_weight: float = 0.8
    llm_weight: float = 0.2

    # 风控（激进风格：放宽单票和杠杆上限）
    max_weight_growth: float = 0.12
    max_weight_bluechip: float = 0.15
    max_weight_leverage: float = 0.10
    max_industry_weight: float = 0.40
    max_leverage_etf_total: float = 0.25

    # 选股阈值（Top-N + 最低分双重机制）
    top_n_buy: int = 12
    min_score_to_buy: float = 55
    min_score_to_hold: float = 45

    # 股票池
    # 成长池：高增速 / 趋势动量强 / 中小至大市值成长股
    growth_tickers: list[str] = field(default_factory=lambda: [
        # 原成长
        "PLTR", "NEOV", "AAOI", "WOLF", "COHR", "NBIS", "CLS", "LITE", "VRT", "GEV",
        # 从蓝筹移入（增速高、波动大、更偏成长逻辑）
        "NVDA", "AMD", "TSLA", "MU",
        # 新增自选
        "CRCL", "CLPT", "NVTS", "SNDK", "CNQ",
        # 新增自选 - 工业/矿业成长
        "ALM", "IPGP",
        # 新增自选 - 半导体封测/自动驾驶/油服
        "AUR", "KLIC", "ASX", "AMKR", "PUMP",
        # 港股成长
        "2706.HK",
    ])
    # 蓝筹池：大市值 / 稳定现金流（蓝筹引擎也会评估成长性）
    bluechip_tickers: list[str] = field(default_factory=lambda: [
        "AAPL", "MSFT", "AMZN", "AVGO", "ORCL", "XLE", "B",
        # 新增自选 - 资源/化工蓝筹
        "SCCO", "ALB",
    ])
    # 港股池（使用蓝筹引擎评估）
    hk_bluechip_tickers: list[str] = field(default_factory=lambda: [
        "0700.HK", "9988.HK",
    ])
    # 杠杆ETF池（纯趋势+波动率引擎）
    leverage_etf_tickers: list[str] = field(default_factory=lambda: [
        "SOXL", "DFEN", "NVTX", "LITX", "SNXX", "7747.HK", "7709.HK",
    ])

    # 杠杆ETF → 底层股票映射（有底层的会引入底层基本面+技术面）
    leverage_underlying_map: dict[str, str] = field(default_factory=lambda: {
        "NVTX": "NVTS",
        "LITX": "LITE",
        "SNXX": "SNDK",
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
        "NVTX": "2倍做多纳微半导体(NVTS)，功率半导体企业，受电动车/新能源需求驱动",
        "LITX": "2倍做多Lumentum(LITE)，光通信和3D传感龙头，受AI数据中心光模块需求驱动",
        "SNXX": "2倍做多闪迪(SNDK)，存储芯片企业，受NAND闪存价格周期和AI存储需求影响",
        "7747.HK": "南方两倍做多纳指ETF，港股杠杆产品，跟踪纳斯达克100指数",
        "7709.HK": "南方两倍做多恒指ETF，港股杠杆产品，跟踪恒生指数",
    })

    sector_map: dict[str, str] = field(default_factory=lambda: {
        # 蓝筹
        "AAPL": "Tech", "MSFT": "Tech", "AVGO": "Tech", "ORCL": "Tech",
        "AMZN": "Consumer", "XLE": "Energy", "B": "Materials",
        "SCCO": "Materials", "ALB": "Materials",
        # 成长
        "NVDA": "Tech", "AMD": "Tech", "MU": "Tech", "TSLA": "Auto",
        "PLTR": "Tech", "AAOI": "Tech", "WOLF": "Tech", "COHR": "Tech",
        "NBIS": "Tech", "CLS": "Tech", "LITE": "Tech",
        "VRT": "Industrials", "GEV": "Industrials", "NEOV": "Energy",
        "CRCL": "Fintech", "CLPT": "Healthcare", "NVTS": "Tech",
        "SNDK": "Tech", "CNQ": "Energy",
        "ALM": "Materials", "IPGP": "Industrials",
        "AUR": "Auto", "KLIC": "Tech", "ASX": "Tech", "AMKR": "Tech", "PUMP": "Energy",
        # 港股
        "0700.HK": "Tech", "9988.HK": "Tech",
        "2706.HK": "Tech",
        # 杠杆ETF
        "SOXL": "Leveraged", "DFEN": "Leveraged",
        "NVTX": "Leveraged", "LITX": "Leveraged", "SNXX": "Leveraged",
        "7747.HK": "Leveraged", "7709.HK": "Leveraged",
    })

    @property
    def all_tickers(self) -> list[str]:
        ordered = (
            self.growth_tickers
            + self.bluechip_tickers
            + self.hk_bluechip_tickers
            + self.leverage_etf_tickers
        )
        # 加入参考指数/ETF，用于杠杆引擎技术面分析
        ref_tickers = list(self.leverage_index_map.values())
        ordered = ordered + ref_tickers
        return list(dict.fromkeys(ordered))


def default_config() -> StrategyConfig:
    return StrategyConfig()
