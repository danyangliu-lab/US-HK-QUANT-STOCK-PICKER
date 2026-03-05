from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

CACHE_DIR = Path("cache")


@dataclass
class MarketData:
    prices: pd.DataFrame
    volumes: pd.DataFrame
    fundamentals: pd.DataFrame
    vix: pd.Series | None = None  # VIX恐慌指数收盘序列


def _safe_float(v: object, default: float = np.nan) -> float:
    if v is None:
        return default
    try:
        return float(v)  # type: ignore[arg-type]
    except Exception:
        return default


def _cache_path(tickers: list[str], label: str) -> Path:
    key = hashlib.md5(",".join(sorted(tickers)).encode()).hexdigest()[:10]
    today = pd.Timestamp.utcnow().strftime("%Y%m%d")
    return CACHE_DIR / f"{label}_{key}_{today}.pkl"


def _load_cache(path: Path) -> pd.DataFrame | None:
    if path.exists():
        try:
            return pd.read_pickle(path)
        except Exception:
            pass
    return None


def _save_cache(df: pd.DataFrame, path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_pickle(path)
    except Exception:
        pass


def download_market_data(tickers: list[str], lookback_days: int = 320) -> MarketData:
    end = pd.Timestamp.utcnow().tz_localize(None)
    start = end - pd.Timedelta(days=max(lookback_days * 2, 365))

    # 行情缓存
    price_cache = _cache_path(tickers, "prices")
    vol_cache = _cache_path(tickers, "volumes")
    cached_prices = _load_cache(price_cache)
    cached_volumes = _load_cache(vol_cache)

    if cached_prices is not None and cached_volumes is not None:
        logger.info("使用今日行情缓存")
        prices = cached_prices
        volumes = cached_volumes
    else:
        raw = yf.download(
            tickers=tickers,
            start=start.strftime("%Y-%m-%d"),
            end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False,
            group_by="ticker",
            threads=True,
        )

        price_map: dict[str, pd.Series] = {}
        vol_map: dict[str, pd.Series] = {}

        failed_tickers: list[str] = []

        for t in tickers:
            try:
                if len(tickers) == 1:
                    px = raw.get("Close")
                    vol = raw.get("Volume")
                else:
                    if t not in raw.columns.get_level_values(0):
                        failed_tickers.append(t)
                        continue
                    px = raw[t].get("Close")
                    vol = raw[t].get("Volume")

                if px is None or len(px.dropna()) == 0:
                    failed_tickers.append(t)
                    continue
                price_map[t] = px.rename(t)
                vol_map[t] = vol.rename(t) if vol is not None else pd.Series(
                    index=px.index, dtype=float, name=t
                )
            except Exception as e:
                logger.warning("解析行情异常 %s: %s", t, e)
                failed_tickers.append(t)

        # 对失败的 ticker 逐只重试（最多2次），使用 yf.Ticker 避免 download 格式问题
        if failed_tickers:
            logger.info("批量下载失败 %d 只，开始逐只重试: %s", len(failed_tickers), failed_tickers)
            for t in failed_tickers:
                for attempt in range(1, 3):
                    try:
                        time.sleep(1)
                        ticker_obj = yf.Ticker(t)
                        hist = ticker_obj.history(
                            start=start.strftime("%Y-%m-%d"),
                            end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                            auto_adjust=True,
                        )
                        if hist is not None and not hist.empty and "Close" in hist.columns:
                            # 去掉时区信息，与 yf.download 的 tz-naive index 保持一致
                            if hist.index.tz is not None:
                                hist.index = hist.index.tz_localize(None)
                            px = hist["Close"].dropna()
                            vol = hist["Volume"] if "Volume" in hist.columns else None
                            if len(px) > 0:
                                price_map[t] = px.rename(t)
                                vol_map[t] = vol.rename(t) if vol is not None else pd.Series(
                                    index=px.index, dtype=float, name=t
                                )
                                logger.info("重试成功 [%d/2]: %s (%d条数据)", attempt, t, len(px))
                                break
                    except Exception as e:
                        logger.warning("重试失败 [%d/2] %s: %s", attempt, t, e)
                else:
                    logger.warning("放弃下载: %s（2次重试均失败）", t)

        prices = pd.concat(price_map.values(), axis=1).sort_index() if price_map else pd.DataFrame()
        volumes = pd.concat(vol_map.values(), axis=1).sort_index() if vol_map else pd.DataFrame()
        logger.info("行情下载完成: 成功 %d / %d", len(price_map), len(tickers))
        _save_cache(prices, price_cache)
        _save_cache(volumes, vol_cache)

    prices = prices.tail(lookback_days).copy()
    volumes = volumes.reindex(prices.index).copy()

    # 基本面缓存
    fund_cache = _cache_path(tickers, "fund")
    cached_fund = _load_cache(fund_cache)
    if cached_fund is not None:
        logger.info("使用今日基本面缓存")
        fundamentals_df = cached_fund
    else:
        fundamentals = []
        for t in prices.columns.tolist():
            info: dict = {}
            try:
                info = yf.Ticker(t).info or {}
            except Exception as e:
                logger.warning("基本面拉取失败 %s: %s", t, e)

            fundamentals.append({
                "ticker": t,
                "market_cap": _safe_float(info.get("marketCap")),
                "trailing_pe": _safe_float(info.get("trailingPE")),
                "forward_pe": _safe_float(info.get("forwardPE")),
                "price_to_sales": _safe_float(info.get("priceToSalesTrailing12Months")),
                "price_to_book": _safe_float(info.get("priceToBook")),
                "gross_margin": _safe_float(info.get("grossMargins")),
                "profit_margin": _safe_float(info.get("profitMargins")),
                "revenue_growth": _safe_float(info.get("revenueGrowth")),
                "earnings_growth": _safe_float(info.get("earningsGrowth")),
                "return_on_equity": _safe_float(info.get("returnOnEquity")),
                "operating_cashflow": _safe_float(info.get("operatingCashflow")),
                "free_cashflow": _safe_float(info.get("freeCashflow")),
                "debt_to_equity": _safe_float(info.get("debtToEquity")),
                "beta": _safe_float(info.get("beta")),
                "average_volume": _safe_float(info.get("averageVolume")),
                "sector": info.get("sector"),
                "short_name": info.get("shortName"),
            })

        fundamentals_df = pd.DataFrame(fundamentals).set_index("ticker") if fundamentals else pd.DataFrame()
        _save_cache(fundamentals_df, fund_cache)

    return MarketData(prices=prices, volumes=volumes, fundamentals=fundamentals_df, vix=_download_vix(prices))


def _download_vix(prices: pd.DataFrame) -> pd.Series | None:
    """下载 VIX 恐慌指数，与行情数据同时间范围对齐。"""
    vix_cache = _cache_path(["^VIX"], "vix")
    cached = _load_cache(vix_cache)
    if cached is not None:
        logger.info("使用今日VIX缓存")
        vix = cached.squeeze()
    else:
        try:
            start = prices.index[0].strftime("%Y-%m-%d")
            end = (prices.index[-1] + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            vix_data = yf.download("^VIX", start=start, end=end, progress=False, auto_adjust=True)
            if vix_data is None or vix_data.empty:
                logger.warning("VIX 数据下载为空")
                return None
            vix = vix_data["Close"].squeeze()
            if isinstance(vix, pd.DataFrame):
                vix = vix.iloc[:, 0]
            vix = vix.dropna()
            _save_cache(vix.to_frame("vix"), vix_cache)
            logger.info("VIX 数据下载完成: %d 条", len(vix))
        except Exception as e:
            logger.warning("VIX 数据下载失败: %s", e)
            return None
    if isinstance(vix, pd.DataFrame):
        vix = vix.iloc[:, 0]
    return vix.reindex(prices.index, method="ffill")
