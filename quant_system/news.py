"""新闻抓取模块。

从海外主流媒体（Google News RSS）获取最新财经/地缘政治新闻，
自动注入到 LLM prompt 中，让大模型基于真实时事做出判断。

数据源：
- Google News RSS（免费，无需 API Key，聚合 Reuters/Bloomberg/CNBC/BBC 等）
- 按关键词搜索，支持全局新闻 + 个股/行业新闻
- 当日缓存，避免重复请求
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

CACHE_DIR = Path("cache")
NEWS_CACHE_FILE = "news_cache.pkl"

# 全局关键词：宏观/地缘政治/美联储/贸易
GLOBAL_KEYWORDS = [
    "stock market today",
    "geopolitics war conflict",
    "Federal Reserve interest rate",
    "US China trade tariff",
    "oil price OPEC",
    "semiconductor chip AI",
]


@dataclass
class NewsContext:
    """新闻上下文，传给 LLM prompt。"""
    global_headlines: list[str] = field(default_factory=list)
    ticker_headlines: dict[str, list[str]] = field(default_factory=dict)
    fetch_time: str = ""


def _fetch_google_news_rss(query: str, max_results: int = 8) -> list[str]:
    """从 Google News RSS 抓取新闻标题。"""
    import xml.etree.ElementTree as ET

    url = f"https://news.google.com/rss/search?q={requests.utils.quote(query)}&hl=en-US&gl=US&ceid=US:en"
    try:
        resp = requests.get(url, timeout=15, headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
        })
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        titles = []
        for item in root.findall(".//item"):
            title_el = item.find("title")
            if title_el is not None and title_el.text:
                titles.append(title_el.text.strip())
            if len(titles) >= max_results:
                break
        return titles
    except Exception as e:
        logger.warning("Google News RSS 抓取失败 [%s]: %s", query, e)
        return []


def _load_news_cache() -> NewsContext | None:
    path = CACHE_DIR / NEWS_CACHE_FILE
    if path.exists():
        try:
            data = pd.read_pickle(path)
            if isinstance(data, NewsContext):
                # 检查是否当日缓存
                today = pd.Timestamp.utcnow().strftime("%Y-%m-%d")
                if data.fetch_time.startswith(today):
                    return data
        except Exception:
            pass
    return None


def _save_news_cache(ctx: NewsContext) -> None:
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        pd.to_pickle(ctx, CACHE_DIR / NEWS_CACHE_FILE)
    except Exception:
        pass


def fetch_news(
    tickers: list[str],
    ticker_names: dict[str, str] | None = None,
    sector_desc: dict[str, str] | None = None,
    max_global: int = 5,
    max_per_ticker: int = 3,
) -> NewsContext:
    """
    抓取全局新闻 + 各标的相关新闻。

    参数:
        tickers: 需要抓取新闻的标的列表
        ticker_names: ticker → 英文名映射（用于搜索）
        sector_desc: ticker → 行业描述映射（杠杆ETF用）
        max_global: 每个全局关键词取的标题数
        max_per_ticker: 每个标的取的标题数
    """
    # 尝试读取当日缓存
    cached = _load_news_cache()
    if cached is not None:
        logger.info("使用今日新闻缓存 (%d条全局, %d个标的)", len(cached.global_headlines), len(cached.ticker_headlines))
        return cached

    logger.info("开始抓取新闻...")
    names = ticker_names or {}
    sdesc = sector_desc or {}

    # 1. 全局新闻
    global_headlines: list[str] = []
    for kw in GLOBAL_KEYWORDS:
        titles = _fetch_google_news_rss(kw, max_results=max_global)
        global_headlines.extend(titles)
        time.sleep(0.3)  # 避免请求过快

    # 去重
    seen = set()
    unique_global = []
    for h in global_headlines:
        if h not in seen:
            seen.add(h)
            unique_global.append(h)
    global_headlines = unique_global[:30]  # 最多保留30条

    logger.info("全局新闻: %d 条", len(global_headlines))

    # 2. 个股/ETF 新闻
    ticker_headlines: dict[str, list[str]] = {}
    for t in tickers:
        # 构建搜索词
        search_terms = []
        name = names.get(t, "")
        if name and name != t:
            search_terms.append(f"{name} stock")
        if t in sdesc:
            # 杠杆ETF：用行业关键词搜索
            desc = sdesc[t]
            if "半导体" in desc or "semiconductor" in desc.lower():
                search_terms.append("semiconductor chip industry news")
            elif "航空国防" in desc or "defense" in desc.lower():
                search_terms.append("defense military geopolitics news")
            elif "纳指" in desc or "nasdaq" in desc.lower():
                search_terms.append("Nasdaq 100 tech stocks news")
            elif "恒指" in desc or "hang seng" in desc.lower():
                search_terms.append("Hong Kong stock market news")
        else:
            # 普通股票：用 ticker + stock 搜索
            search_terms.append(f"{t} stock news")

        all_titles: list[str] = []
        for term in search_terms[:2]:  # 每个标的最多2个搜索词
            titles = _fetch_google_news_rss(term, max_results=max_per_ticker)
            all_titles.extend(titles)
            time.sleep(0.3)

        if all_titles:
            ticker_headlines[t] = all_titles[:max_per_ticker * 2]

    logger.info("个股新闻: 覆盖 %d / %d 个标的", len(ticker_headlines), len(tickers))

    ctx = NewsContext(
        global_headlines=global_headlines,
        ticker_headlines=ticker_headlines,
        fetch_time=pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
    )
    _save_news_cache(ctx)
    return ctx


def format_news_for_prompt(
    ticker: str,
    news_ctx: NewsContext,
    max_global: int = 10,
    max_ticker: int = 5,
) -> str:
    """将新闻格式化为 LLM prompt 中的文本片段。"""
    parts: list[str] = []

    # 全局新闻
    if news_ctx.global_headlines:
        headlines = news_ctx.global_headlines[:max_global]
        parts.append("【近期全球财经/地缘政治要闻】")
        for i, h in enumerate(headlines, 1):
            parts.append(f"{i}. {h}")

    # 个股新闻
    ticker_news = news_ctx.ticker_headlines.get(ticker, [])
    if ticker_news:
        parts.append(f"\n【{ticker} 相关新闻】")
        for i, h in enumerate(ticker_news[:max_ticker], 1):
            parts.append(f"{i}. {h}")

    if not parts:
        return ""

    return "\n".join(parts)
