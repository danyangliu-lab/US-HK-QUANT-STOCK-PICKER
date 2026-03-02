from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, TypedDict

import pandas as pd
import requests

logger = logging.getLogger(__name__)


class EventResult(TypedDict):
    event_score: float
    risk_flag: int
    reason: str


@dataclass
class LLMConfig:
    enabled: bool = False
    base_url: str = "https://api.hunyuan.cloud.tencent.com/v1"
    api_key: str = ""
    model: str = "hunyuan-2.0-thinking-20251109"
    timeout_sec: int = 30

    secret_id: str = ""
    secret_key: str = ""
    region: str = "ap-guangzhou"

    @classmethod
    def from_env(cls) -> LLMConfig:
        enabled = os.getenv("HUNYUAN_ENABLED", "false").lower() in {"1", "true", "yes"}
        return cls(
            enabled=enabled,
            base_url=os.getenv("HUNYUAN_BASE_URL", "https://api.hunyuan.cloud.tencent.com/v1"),
            api_key=os.getenv("HUNYUAN_API_KEY", ""),
            model=os.getenv("HUNYUAN_MODEL", "hunyuan-2.0-thinking-20251109"),
            timeout_sec=int(os.getenv("HUNYUAN_TIMEOUT", "30")),
            secret_id=os.getenv("TENCENT_SECRET_ID", ""),
            secret_key=os.getenv("TENCENT_SECRET_KEY", ""),
            region=os.getenv("TENCENT_REGION", "ap-guangzhou"),
        )


def _to_float(value: object, default: float = 0.0) -> float:
    if isinstance(value, (int, float, str, bool)):
        try:
            return float(value)
        except Exception:
            return default
    return default


def _build_prompt(
    ticker: str,
    quant_row: Mapping[str, object],
    fundamentals: Mapping[str, object],
    sector_desc: str = "",
    news_text: str = "",
) -> str:
    annual_vol = _to_float(quant_row.get("annual_vol", 0.0))
    engine = str(quant_row.get("engine", "unknown"))

    short_name = str(fundamentals.get("short_name", ticker))
    sector = str(fundamentals.get("sector", "Unknown"))
    rev_growth = _to_float(fundamentals.get("revenue_growth", 0.0))
    gross_margin = _to_float(fundamentals.get("gross_margin", 0.0))
    market_cap = _to_float(fundamentals.get("market_cap", 0.0))

    # 新闻片段（如果有）
    news_block = ""
    if news_text:
        news_block = f"\n\n{news_text}\n"

    # 杠杆ETF用定制prompt
    if engine == "leverage" and sector_desc:
        return (
            "你是美股/港股量化研究员，擅长分析杠杆ETF和宏观/行业趋势。\n"
            "下面提供了从主流媒体抓取的最新新闻标题，请结合这些真实新闻进行分析。\n"
            "给出独立的事件评分event_score（-1到1，正=利好，负=利空）和风险标记risk_flag（0=正常，1=有近期风险）。\n"
            "注意：请完全基于新闻/事件/宏观因素做独立判断，不要参考任何量化分数。\n"
            "仅输出JSON，不要解释。\n\n"
            f"ETF: {ticker} ({short_name})\n"
            f"产品描述: {sector_desc}\n"
            f"年化波动率: {annual_vol:.2%}\n"
            f"{news_block}"
            "请重点考虑:\n"
            "1. 上述新闻中与该ETF行业直接相关的事件及其影响方向\n"
            "2. 地缘政治冲突、战争、制裁对该行业的影响\n"
            "3. 宏观经济环境（利率、通胀、贸易政策）对该行业的影响\n"
            "4. 杠杆ETF特有风险（波动率衰减、持仓成本）\n"
            '输出格式: {"event_score":0.12,"risk_flag":0,"reason":"简短中文原因"}'
        )

    return (
        "你是美股/港股量化研究员，擅长分析个股基本面、行业趋势和时事新闻。\n"
        "下面提供了从主流媒体抓取的最新新闻标题，请结合这些真实新闻进行分析。\n"
        "给出独立的事件评分event_score（-1到1，正=利好，负=利空）和风险标记risk_flag（0=正常，1=有近期风险）。\n"
        "注意：请完全基于新闻/事件/宏观因素做独立判断，不要参考任何量化分数。\n"
        "仅输出JSON，不要解释。\n\n"
        f"股票: {ticker} ({short_name})\n"
        f"行业: {sector}\n"
        f"市值: {market_cap/1e9:.1f}B USD\n"
        f"营收增长: {rev_growth:.1%}, 毛利率: {gross_margin:.1%}\n"
        f"年化波动率: {annual_vol:.2%}\n"
        f"{news_block}"
        "请重点考虑:\n"
        "1. 上述新闻中与该公司/行业直接相关的事件及其影响\n"
        "2. 地缘政治、战争、制裁等对该公司/行业的影响\n"
        "3. 宏观经济环境（利率、通胀、贸易政策）的影响\n"
        "4. 近期财报、并购、监管风险等公司层面事件\n"
        '输出格式: {"event_score":0.12,"risk_flag":0,"reason":"简短中文原因"}'
    )


def _fallback(reason: str) -> EventResult:
    return {"event_score": 0.0, "risk_flag": 0, "reason": reason}


def _parse_json_safe(text: str) -> EventResult:
    payload = text.strip()
    if payload.startswith("```"):
        payload = payload.strip("`")
        payload = payload.replace("json", "", 1).strip()
    # 兼容 thinking 模型可能输出 <think>...</think>JSON 的情况
    if "<think>" in payload:
        idx = payload.rfind("}")
        if idx >= 0:
            start = payload.find("{")
            if start >= 0:
                payload = payload[start : idx + 1]
    try:
        data = json.loads(payload)
        event_score = _to_float(data.get("event_score", 0.0))
        event_score = max(min(event_score, 1.0), -1.0)
        risk_flag = 1 if int(_to_float(data.get("risk_flag", 0))) == 1 else 0
        reason = str(data.get("reason", ""))[:120]
        return {"event_score": event_score, "risk_flag": risk_flag, "reason": reason}
    except Exception:
        return _fallback("parse_failed")


def _call_chat_openai_compatible(cfg: LLMConfig, prompt: str) -> EventResult:
    url = cfg.base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {cfg.api_key}",
        "Content-Type": "application/json",
    }
    payload: dict[str, Any] = {
        "model": cfg.model,
        "messages": [
            {"role": "system", "content": "你是严格JSON输出助手。仅输出JSON，无其他文字。"},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
    }
    r = requests.post(url, headers=headers, json=payload, timeout=cfg.timeout_sec)
    r.raise_for_status()
    data = r.json()
    choices = data.get("choices", [])
    if not choices:
        return _fallback("empty_choices")
    message = choices[0].get("message", {})
    content = str(message.get("content", ""))
    return _parse_json_safe(content)


def _call_chat_tencent_sdk(cfg: LLMConfig, prompt: str) -> EventResult:
    try:
        import importlib
        credential_mod = importlib.import_module("tencentcloud.common.credential")
        hunyuan_client_mod = importlib.import_module("tencentcloud.hunyuan.v20230901.hunyuan_client")
        models_mod = importlib.import_module("tencentcloud.hunyuan.v20230901.models")
    except Exception:
        return _fallback("sdk_not_installed")

    try:
        cred = credential_mod.Credential(cfg.secret_id, cfg.secret_key)
        client = hunyuan_client_mod.HunyuanClient(cred, cfg.region)

        req = models_mod.ChatCompletionsRequest()
        req.from_json_string(json.dumps({
            "Model": cfg.model,
            "Messages": [
                {"Role": "system", "Content": "你是严格JSON输出助手。仅输出JSON，无其他文字。"},
                {"Role": "user", "Content": prompt},
            ],
            "Stream": False,
        }, ensure_ascii=False))

        resp = client.ChatCompletions(req)
        data = json.loads(resp.to_json_string())

        choices = data.get("Choices") or data.get("choices") or []
        if not choices:
            return _fallback("empty_choices")

        message = choices[0].get("Message") or choices[0].get("message") or {}
        content = str(message.get("Content") or message.get("content") or "")
        if not content:
            return _fallback("empty_content")
        return _parse_json_safe(content)
    except Exception as e:
        logger.warning("SDK调用异常: %s", e)
        return _fallback("sdk_call_error")


def _call_chat(cfg: LLMConfig, prompt: str) -> EventResult:
    if cfg.secret_id and cfg.secret_key:
        return _call_chat_tencent_sdk(cfg, prompt)
    if cfg.api_key:
        return _call_chat_openai_compatible(cfg, prompt)
    return _fallback("missing_credentials")


def batch_event_score(
    tickers: list[str],
    quant_table: pd.DataFrame,
    fundamentals: pd.DataFrame,
    cfg: LLMConfig,
    sector_desc_map: dict[str, str] | None = None,
    news_ctx: object | None = None,
) -> dict[str, EventResult]:
    out: dict[str, EventResult] = {}
    sdmap = sector_desc_map or {}

    # 新闻格式化函数（延迟导入避免循环依赖）
    _format_news: Any = None
    if news_ctx is not None:
        try:
            from .news import format_news_for_prompt
            _format_news = format_news_for_prompt
        except ImportError:
            pass

    if not cfg.enabled:
        for t in tickers:
            out[t] = _fallback("llm_disabled")
        return out

    logger.info("========== 开始调用大模型 (模型: %s, 共%d只股票, 新闻: %s) ==========",
                cfg.model, len(tickers), "已注入" if news_ctx else "无")
    success_count = 0
    fail_count = 0

    for i, t in enumerate(tickers, 1):
        row = quant_table.loc[t].to_dict()
        fund_row = fundamentals.loc[t].to_dict() if t in fundamentals.index else {}
        news_text = _format_news(t, news_ctx) if _format_news and news_ctx else ""
        prompt = _build_prompt(t, row, fund_row, sector_desc=sdmap.get(t, ""), news_text=news_text)
        try:
            result = _call_chat(cfg, prompt)
            if result["reason"] in ("sdk_not_installed", "missing_credentials", "sdk_call_error", "parse_failed", "empty_choices", "empty_content"):
                fail_count += 1
                logger.warning("LLM [%d/%d] %s 调用失败: %s", i, len(tickers), t, result["reason"])
            else:
                success_count += 1
                logger.info("LLM [%d/%d] %s 调用成功: score=%.2f flag=%d reason=%s", i, len(tickers), t, result["event_score"], result["risk_flag"], result["reason"])
            out[t] = result
        except Exception as e:
            fail_count += 1
            logger.warning("LLM [%d/%d] %s 异常: %s", i, len(tickers), t, e)
            out[t] = _fallback("llm_error")

    logger.info("========== 大模型调用完成: 成功 %d 只, 失败 %d 只 ==========", success_count, fail_count)
    return out
