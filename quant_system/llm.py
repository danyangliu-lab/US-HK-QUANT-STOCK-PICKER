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
    """三模型交叉验证配置（Kimi K2.5 + DeepSeek V3.2 + GLM-5）。

    三个模型均通过腾讯云 lkeap OpenAI 兼容接口（/v3）调用，
    共用同一个 API Key。
    """
    enabled: bool = False

    # ── 腾讯云 lkeap 统一配置 ──
    lkeap_base_url: str = "https://api.lkeap.cloud.tencent.com/v3"
    lkeap_api_key: str = ""                # 控制台 API Key（三模型共用）
    # 兼容旧版 SDK 调用 DeepSeek 的腾讯云密钥
    secret_id: str = ""
    secret_key: str = ""
    region: str = "ap-guangzhou"

    # ── Kimi K2.5（主模型 A）──
    kimi_enabled: bool = True
    kimi_model: str = "kimi-k2.5"
    kimi_timeout_sec: int = 90

    # ── DeepSeek V3.2（主模型 B）──
    deepseek_enabled: bool = True
    deepseek_model: str = "deepseek-v3.2"
    deepseek_enable_search: bool = True    # DeepSeek 联网搜索（lkeap SDK）
    deepseek_timeout_sec: int = 90
    # 兼容旧版：如有专用 DeepSeek API Key 则走 /v1
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.lkeap.cloud.tencent.com/v1"

    # ── GLM-5（主模型 C）──
    glm_enabled: bool = True
    glm_model: str = "glm-5"
    glm_timeout_sec: int = 90

    # ── 交叉验证策略 ──
    # "cross"   = 三模型交叉验证（默认）：方向一致增强，分歧取保守
    # "avg"     = 简单平均
    # "primary" = 仅第一个可用模型
    cross_validation_mode: str = "cross"

    @property
    def kimi_available(self) -> bool:
        return self.kimi_enabled and bool(self.lkeap_api_key)

    @property
    def deepseek_available(self) -> bool:
        return self.deepseek_enabled and (
            bool(self.lkeap_api_key)
            or bool(self.deepseek_api_key)
            or (bool(self.secret_id) and bool(self.secret_key))
        )

    @property
    def glm_available(self) -> bool:
        return self.glm_enabled and bool(self.lkeap_api_key)

    @property
    def available_models(self) -> list[str]:
        """返回当前可用的模型名称列表。"""
        models: list[str] = []
        if self.kimi_available:
            models.append("kimi")
        if self.deepseek_available:
            models.append("deepseek")
        if self.glm_available:
            models.append("glm")
        return models

    @property
    def multi_model(self) -> bool:
        """是否有 ≥2 个模型可用（可做交叉验证）。"""
        return len(self.available_models) >= 2

    @classmethod
    def from_env(cls) -> LLMConfig:
        enabled = os.getenv("LLM_ENABLED", os.getenv("HUNYUAN_ENABLED", "false")).lower() in {"1", "true", "yes"}

        lkeap_api_key = os.getenv("LKEAP_API_KEY", "")

        kimi_enabled = os.getenv("KIMI_ENABLED", "true").lower() in {"1", "true", "yes"}
        deepseek_enabled = os.getenv("DEEPSEEK_ENABLED", "true").lower() in {"1", "true", "yes"}
        glm_enabled = os.getenv("GLM_ENABLED", "true").lower() in {"1", "true", "yes"}

        return cls(
            enabled=enabled,
            lkeap_base_url=os.getenv("LKEAP_BASE_URL", "https://api.lkeap.cloud.tencent.com/v3"),
            lkeap_api_key=lkeap_api_key,
            secret_id=os.getenv("TENCENT_SECRET_ID", ""),
            secret_key=os.getenv("TENCENT_SECRET_KEY", ""),
            region=os.getenv("TENCENT_REGION", "ap-guangzhou"),
            kimi_enabled=kimi_enabled,
            kimi_model=os.getenv("KIMI_MODEL", "kimi-k2.5"),
            kimi_timeout_sec=int(os.getenv("KIMI_TIMEOUT", "90")),
            deepseek_enabled=deepseek_enabled,
            deepseek_model=os.getenv("DEEPSEEK_MODEL", "deepseek-v3.2"),
            deepseek_enable_search=os.getenv("DEEPSEEK_ENABLE_SEARCH", "true").lower() in {"1", "true", "yes"},
            deepseek_timeout_sec=int(os.getenv("DEEPSEEK_TIMEOUT", "90")),
            deepseek_api_key=os.getenv("DEEPSEEK_API_KEY", ""),
            deepseek_base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.lkeap.cloud.tencent.com/v1"),
            glm_enabled=glm_enabled,
            glm_model=os.getenv("GLM_MODEL", "glm-5"),
            glm_timeout_sec=int(os.getenv("GLM_TIMEOUT", "90")),
            cross_validation_mode=os.getenv("CROSS_VALIDATION_MODE", "cross").lower(),
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
    annual_vol_raw = quant_row.get("annual_vol", None)
    if isinstance(annual_vol_raw, (dict, pd.Series)):
        try:
            annual_vol_raw = next(iter(annual_vol_raw.values())) if isinstance(annual_vol_raw, dict) else annual_vol_raw.iloc[0]
        except Exception:
            annual_vol_raw = None
    annual_vol_str = f"{float(annual_vol_raw):.2%}" if annual_vol_raw is not None and pd.notna(annual_vol_raw) else "数据不足"
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
            f"年化波动率: {annual_vol_str}\n"
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
        f"年化波动率: {annual_vol_str}\n"
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


# ── 通用 lkeap /v3 OpenAI 兼容调用 ──────────────────────

def _call_lkeap_openai(
    cfg: LLMConfig, model: str, prompt: str, timeout: int,
    *, system_msg: str = "你是严格JSON输出助手。仅输出JSON，无其他文字。",
    temperature: float = 0.1,
) -> str:
    """通过腾讯云 lkeap /v3 OpenAI 兼容接口调用任意模型，返回原始文本。"""
    url = cfg.lkeap_base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {cfg.lkeap_api_key}",
        "Content-Type": "application/json",
    }
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ],
    }
    # Kimi K2.5 不支持 temperature/top_p 等采样参数，传入会 400
    if not model.startswith("kimi"):
        payload["temperature"] = temperature
    # Kimi/GLM thinking 模式默认开启，无需显式传参
    r = requests.post(url, headers=headers, json=payload, timeout=timeout)
    if r.status_code != 200:
        logger.warning("lkeap /v3 [%s] HTTP %d: %s", model, r.status_code, r.text[:300])
    r.raise_for_status()
    data = r.json()
    choices = data.get("choices", [])
    if not choices:
        return ""
    message = choices[0].get("message", {})
    return str(message.get("content", ""))


# ── Kimi K2.5 调用 ──────────────────────────────────────

def _call_kimi(cfg: LLMConfig, prompt: str) -> EventResult:
    """通过 lkeap /v3 调用 Kimi K2.5。"""
    raw = _call_kimi_raw(cfg, prompt)
    if not raw:
        return _fallback("kimi_empty")
    return _parse_json_safe(raw)


def _call_kimi_raw(cfg: LLMConfig, prompt: str) -> str:
    if not cfg.kimi_available:
        return ""
    try:
        return _call_lkeap_openai(cfg, cfg.kimi_model, prompt, cfg.kimi_timeout_sec)
    except Exception as e:
        logger.warning("Kimi调用异常: %s: %s", type(e).__name__, e)
        return ""


# ── DeepSeek V3.2 调用 ──────────────────────────────────

def _call_deepseek(cfg: LLMConfig, prompt: str) -> EventResult:
    """调用 DeepSeek：优先 lkeap /v3，其次专用 API Key /v1，最后 SDK。"""
    raw = _call_deepseek_raw(cfg, prompt)
    if not raw:
        return _fallback("ds_empty")
    return _parse_json_safe(raw)


def _call_deepseek_raw(cfg: LLMConfig, prompt: str) -> str:
    """DeepSeek 原始文本：优先 lkeap /v3，其次专用 API Key /v1，最后 SDK。"""
    # 路径1：lkeap /v3 统一接口
    if cfg.lkeap_api_key:
        try:
            return _call_lkeap_openai(cfg, cfg.deepseek_model, prompt, cfg.deepseek_timeout_sec)
        except Exception as e:
            logger.warning("DeepSeek lkeap/v3 异常: %s，尝试备用路径", e)
    # 路径2：专用 API Key + /v1
    if cfg.deepseek_api_key:
        return _call_deepseek_raw_v1(cfg, prompt)
    # 路径3：SDK
    if cfg.secret_id and cfg.secret_key:
        return _call_deepseek_raw_sdk(cfg, prompt)
    return ""


def _call_deepseek_raw_v1(cfg: LLMConfig, prompt: str) -> str:
    """通过 lkeap /v1 专用 API Key 调用 DeepSeek。"""
    url = cfg.deepseek_base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {cfg.deepseek_api_key}",
        "Content-Type": "application/json",
    }
    payload: dict[str, Any] = {
        "model": cfg.deepseek_model,
        "messages": [
            {"role": "system", "content": "你是严格JSON输出助手。仅输出JSON，无其他文字。"},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
    }
    if cfg.deepseek_enable_search:
        payload["enable_search"] = True
    r = requests.post(url, headers=headers, json=payload, timeout=cfg.deepseek_timeout_sec)
    r.raise_for_status()
    data = r.json()
    choices = data.get("choices", [])
    if not choices:
        return ""
    message = choices[0].get("message", {})
    return str(message.get("content", ""))


def _call_deepseek_raw_sdk(cfg: LLMConfig, prompt: str) -> str:
    """通过腾讯云 lkeap SDK（CommonClient + SecretId/SecretKey）获取 DeepSeek 原始响应。"""
    try:
        import importlib
        credential_mod = importlib.import_module("tencentcloud.common.credential")
        common_client_mod = importlib.import_module("tencentcloud.common.common_client")
        http_profile_mod = importlib.import_module("tencentcloud.common.profile.http_profile")
        client_profile_mod = importlib.import_module("tencentcloud.common.profile.client_profile")
    except Exception:
        return '{"action":"持有","confidence":0.0,"reason":"ds_sdk_not_installed"}'

    try:
        cred = credential_mod.Credential(cfg.secret_id, cfg.secret_key)

        http_profile = http_profile_mod.HttpProfile()
        http_profile.endpoint = "lkeap.tencentcloudapi.com"
        http_profile.reqTimeout = cfg.deepseek_timeout_sec

        client_profile = client_profile_mod.ClientProfile()
        client_profile.httpProfile = http_profile

        params: dict[str, Any] = {
            "Model": cfg.deepseek_model,
            "Messages": [
                {"Role": "system", "Content": "你是严格JSON输出助手。仅输出JSON，无其他文字。"},
                {"Role": "user", "Content": prompt},
            ],
            "Stream": False,
        }
        if cfg.deepseek_enable_search:
            params["EnableSearch"] = True

        # 非流式响应辅助类
        class _NonStreamResp:
            def __init__(self) -> None:
                self.response = ""
            def _deserialize(self, obj: object) -> None:
                self.response = json.dumps(obj, ensure_ascii=False)

        common_client = common_client_mod.CommonClient(
            "lkeap", "2024-05-22", cred, cfg.region, profile=client_profile,
        )
        resp = common_client._call_and_deserialize("ChatCompletions", params, _NonStreamResp)

        data = json.loads(resp.response) if isinstance(resp.response, str) else {}
        response_body = data.get("Response", data)
        choices = response_body.get("Choices") or response_body.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("Message") or choices[0].get("message") or {}
        return str(message.get("Content") or message.get("content") or "")
    except Exception as e:
        logger.warning("DeepSeek SDK调用异常: %s: %s", type(e).__name__, e)
        return f'{{"action":"持有","confidence":0.0,"reason":"ds_sdk_error: {e}"}}'


# ── GLM-5 调用 ──────────────────────────────────────────

def _call_glm(cfg: LLMConfig, prompt: str) -> EventResult:
    """通过 lkeap /v3 调用 GLM-5。"""
    raw = _call_glm_raw(cfg, prompt)
    if not raw:
        return _fallback("glm_empty")
    return _parse_json_safe(raw)


def _call_glm_raw(cfg: LLMConfig, prompt: str) -> str:
    if not cfg.glm_available:
        return ""
    try:
        return _call_lkeap_openai(cfg, cfg.glm_model, prompt, cfg.glm_timeout_sec)
    except Exception as e:
        logger.warning("GLM调用异常: %s: %s", type(e).__name__, e)
        return ""


# ── 统一模型调度 ──────────────────────────────────────────

_MODEL_LABEL = {"kimi": "Kimi K2.5", "deepseek": "DeepSeek V3.2", "glm": "GLM-5"}

_CALL_EVENT: dict[str, Any] = {}   # 延迟填充，避免前向引用


def _init_call_maps() -> None:
    global _CALL_EVENT
    if _CALL_EVENT:
        return
    _CALL_EVENT.update({
        "kimi": _call_kimi,
        "deepseek": _call_deepseek,
        "glm": _call_glm,
    })


_CALL_RAW: dict[str, Any] = {}


def _init_raw_maps() -> None:
    global _CALL_RAW
    if _CALL_RAW:
        return
    _CALL_RAW.update({
        "kimi": _call_kimi_raw,
        "deepseek": _call_deepseek_raw,
        "glm": _call_glm_raw,
    })


def _call_model_event(cfg: LLMConfig, model_name: str, prompt: str) -> EventResult:
    """按模型名调用，返回 EventResult。"""
    _init_call_maps()
    fn = _CALL_EVENT.get(model_name)
    if fn is None:
        return _fallback(f"unknown_model_{model_name}")
    return fn(cfg, prompt)


def _call_model_raw(cfg: LLMConfig, model_name: str, prompt: str) -> str:
    """按模型名调用，返回原始文本。"""
    _init_raw_maps()
    fn = _CALL_RAW.get(model_name)
    if fn is None:
        return ""
    return fn(cfg, prompt)


# ── 三模型交叉验证融合 ──────────────────────────────────

def _cross_validate_event(
    results: dict[str, EventResult],
    mode: str = "cross",
) -> EventResult:
    """
    将多个模型的 EventResult 做交叉验证融合。

    results: {"kimi": EventResult, "deepseek": EventResult, "glm": EventResult}
    mode:
      - "cross"   : 多数方向一致 → 增强信心，分歧 → 取保守值
      - "avg"     : 简单平均
      - "primary" : 仅返回第一个可用模型结果
    """
    if not results:
        return _fallback("no_model_results")

    names = list(results.keys())
    items = list(results.values())

    if mode == "primary" or len(items) == 1:
        return items[0]

    scores = [r["event_score"] for r in items]
    flags = [r["risk_flag"] for r in items]
    reasons = [r["reason"] for r in items]
    merged_flag = 1 if any(f == 1 for f in flags) else 0

    if mode == "avg":
        avg_score = sum(scores) / len(scores)
        label_parts = " | ".join(f"[{_MODEL_LABEL.get(n, n)}] {reasons[i]}" for i, n in enumerate(names))
        return {
            "event_score": max(min(avg_score, 1.0), -1.0),
            "risk_flag": merged_flag,
            "reason": label_parts,
        }

    # mode == "cross" (默认)
    n_positive = sum(1 for s in scores if s >= 0)
    n_negative = len(scores) - n_positive

    # 多数方向
    majority_positive = n_positive > n_negative
    majority_count = max(n_positive, n_negative)
    agreement_ratio = majority_count / len(scores)  # e.g. 3/3=1.0, 2/3=0.67

    if agreement_ratio >= 0.67:
        # 多数一致（2/3 或 3/3）→ 加权平均，一致度越高越大胆
        fused_score = sum(scores) / len(scores)
        if agreement_ratio == 1.0:
            # 完全一致：增强 15%
            fused_score *= 1.15
            # 多模型都给出强信号时额外增强
            if all(abs(s) > 0.3 for s in scores):
                fused_score *= 1.05
            agreement = "全体一致看多" if majority_positive else "全体一致看空"
        else:
            # 2/3 一致：温和增强 5%
            fused_score *= 1.05
            agreement = "多数看多" if majority_positive else "多数看空"
    else:
        # 严重分歧（不太可能发生在3模型中，但兜底）→ 取绝对值最小的，打折
        min_abs_score = min(scores, key=abs)
        fused_score = min_abs_score * 0.70
        agreement = "分歧"

    fused_score = max(min(fused_score, 1.0), -1.0)

    score_parts = " ".join(f"{_MODEL_LABEL.get(n, n)[:2]}:{scores[i]:+.2f}" for i, n in enumerate(names))
    return {
        "event_score": fused_score,
        "risk_flag": merged_flag,
        "reason": f"[{agreement}] {score_parts} | {reasons[0]}",
    }


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

    models = cfg.available_models
    if not models:
        for t in tickers:
            out[t] = _fallback("no_model_available")
        return out

    n_models = len(models)
    mode_desc = f"{n_models}模型交叉验证({cfg.cross_validation_mode})" if n_models > 1 else f"单模型({models[0]})"
    model_names = " + ".join(_MODEL_LABEL.get(m, m) for m in models)

    logger.info("========== 开始调用大模型 (%s: %s, 共%d只股票, 新闻: %s) ==========",
                mode_desc, model_names, len(tickers), "已注入" if news_ctx else "无")
    if n_models > 1:
        logger.info("交叉验证模式: %s", cfg.cross_validation_mode)

    # 统计
    model_success: dict[str, int] = {m: 0 for m in models}
    model_fail: dict[str, int] = {m: 0 for m in models}

    import time as _time
    _RETRY_REASONS = {"sdk_call_error", "empty_choices", "empty_content", "parse_failed",
                      "kimi_empty", "ds_empty", "glm_empty"}

    for i, t in enumerate(tickers, 1):
        row = quant_table.loc[t].to_dict()
        fund_row = fundamentals.loc[t].to_dict() if t in fundamentals.index else {}
        news_text = _format_news(t, news_ctx) if _format_news and news_ctx else ""
        prompt = _build_prompt(t, row, fund_row, sector_desc=sdmap.get(t, ""), news_text=news_text)

        results: dict[str, EventResult] = {}

        for m in models:
            label = _MODEL_LABEL.get(m, m)
            result: EventResult | None = None
            for attempt in range(2):
                try:
                    result = _call_model_event(cfg, m, prompt)
                    if result["reason"] in _RETRY_REASONS and attempt == 0:
                        logger.info("LLM [%d/%d] %s %s首次失败(%s)，2秒后重试...",
                                    i, len(tickers), t, label, result["reason"])
                        _time.sleep(2)
                        continue
                    break
                except Exception as e:
                    if attempt == 0:
                        _time.sleep(2)
                        continue
                    result = _fallback(f"{m}_error")
                    break
            assert result is not None

            fail_reasons = {"kimi_empty", "ds_empty", "glm_empty", "ds_no_credentials",
                            "sdk_not_installed", "missing_credentials", "sdk_call_error",
                            "parse_failed", "empty_choices", "empty_content", f"{m}_error",
                            f"unknown_model_{m}"}
            if result["reason"] in fail_reasons:
                model_fail[m] += 1
                logger.warning("LLM [%d/%d] %s %s调用失败: %s", i, len(tickers), t, label, result["reason"])
            else:
                model_success[m] += 1
                results[m] = result
                logger.info("LLM [%d/%d] %s %s: score=%.2f flag=%d reason=%s",
                            i, len(tickers), t, label, result["event_score"], result["risk_flag"], result["reason"])

        # 融合
        if not results:
            out[t] = _fallback("all_models_failed")
        elif len(results) == 1:
            out[t] = list(results.values())[0]
        else:
            fused = _cross_validate_event(results, cfg.cross_validation_mode)
            logger.info("LLM [%d/%d] %s 融合: score=%.2f flag=%d (%s)",
                        i, len(tickers), t, fused["event_score"], fused["risk_flag"], fused["reason"][:80])
            out[t] = fused

        _time.sleep(0.3)

    stat_parts = ", ".join(f"{_MODEL_LABEL.get(m, m)}成功{model_success[m]}/失败{model_fail[m]}" for m in models)
    logger.info("========== 大模型调用完成 (%s): %s ==========", mode_desc, stat_parts)
    return out


# ── 持仓投资建议 ──────────────────────────────────────


class AdviceResult(TypedDict):
    action: str          # 加仓 / 持有 / 减仓 / 清仓
    confidence: float    # 0~1
    reason: str


def _build_advice_prompt(
    ticker: str,
    quant_row: Mapping[str, object],
    fundamentals: Mapping[str, object],
    sector_desc: str = "",
    news_text: str = "",
) -> str:
    annual_vol_raw = quant_row.get("annual_vol", None)
    if isinstance(annual_vol_raw, (dict, pd.Series)):
        try:
            annual_vol_raw = next(iter(annual_vol_raw.values())) if isinstance(annual_vol_raw, dict) else annual_vol_raw.iloc[0]
        except Exception:
            annual_vol_raw = None
    annual_vol_str = f"{float(annual_vol_raw):.2%}" if annual_vol_raw is not None and pd.notna(annual_vol_raw) else "数据不足"
    engine = str(quant_row.get("engine", "unknown"))
    quant_score = _to_float(quant_row.get("quant_score", 0.0))
    trend_score = _to_float(quant_row.get("trend_score", 0.0))
    final_score = _to_float(quant_row.get("final_score", 0.0))
    action = str(quant_row.get("action", ""))

    short_name = str(fundamentals.get("short_name", ticker))
    sector = str(fundamentals.get("sector", "Unknown"))
    rev_growth = _to_float(fundamentals.get("revenue_growth", 0.0))
    gross_margin = _to_float(fundamentals.get("gross_margin", 0.0))
    market_cap = _to_float(fundamentals.get("market_cap", 0.0))
    trailing_pe = _to_float(fundamentals.get("trailing_pe", 0.0))
    forward_pe = _to_float(fundamentals.get("forward_pe", 0.0))

    news_block = f"\n{news_text}\n" if news_text else ""

    if engine == "leverage" and sector_desc:
        return (
            "你是资深美股/港股投资顾问。我当前持有以下杠杆ETF，请给出具体操作建议。\n"
            "请结合实时新闻、量化评分、行业趋势综合判断。\n\n"
            f"ETF: {ticker} ({short_name})\n"
            f"产品描述: {sector_desc}\n"
            f"量化趋势分: {trend_score:.1f}, 量化总分: {quant_score:.1f}, 最终得分: {final_score:.1f}\n"
            f"系统信号: {action}\n"
            f"年化波动率: {annual_vol_str}\n"
            f"{news_block}\n"
            "请给出操作建议（加仓/持有/减仓/清仓），信心度(0~1)，和详细理由（中文，2-3句话）。\n"
            "特别注意杠杆ETF的波动率衰减风险和持仓成本。\n"
            '仅输出JSON: {"action":"持有","confidence":0.7,"reason":"..."}'
        )

    return (
        "你是资深美股/港股投资顾问。我当前持有以下股票，请给出具体操作建议。\n"
        "请结合实时新闻、量化评分、基本面数据综合判断。\n\n"
        f"股票: {ticker} ({short_name})\n"
        f"行业: {sector}, 市值: {market_cap/1e9:.1f}B USD\n"
        f"营收增长: {rev_growth:.1%}, 毛利率: {gross_margin:.1%}\n"
        f"Trailing PE: {trailing_pe:.1f}, Forward PE: {forward_pe:.1f}\n"
        f"量化趋势分: {trend_score:.1f}, 量化总分: {quant_score:.1f}, 最终得分: {final_score:.1f}\n"
        f"系统信号: {action}\n"
        f"年化波动率: {annual_vol_str}\n"
        f"{news_block}\n"
        "请给出操作建议（加仓/持有/减仓/清仓），信心度(0~1)，和详细理由（中文，2-3句话）。\n"
        '仅输出JSON: {"action":"持有","confidence":0.7,"reason":"..."}'
    )


def _parse_advice_safe(text: str) -> AdviceResult:
    payload = text.strip()
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
        action = str(data.get("action", "持有"))
        confidence = max(min(_to_float(data.get("confidence", 0.5)), 1.0), 0.0)
        reason = str(data.get("reason", ""))[:200]
        return {"action": action, "confidence": confidence, "reason": reason}
    except Exception:
        return {"action": "持有", "confidence": 0.0, "reason": "解析失败"}


# ── 投资建议交叉验证 ──────────────────────────────────

_ACTION_RANK = {"清仓": 0, "减仓": 1, "持有": 2, "加仓": 3}
_RANK_ACTION = {v: k for k, v in _ACTION_RANK.items()}


def _cross_validate_advice(
    results: dict[str, AdviceResult],
    mode: str = "cross",
) -> AdviceResult:
    """
    三模型交叉验证投资建议。

    mode:
      - "cross"   : 一致时增强信心，分歧时取保守建议
      - "avg"     : 平均信心，取多数动作
      - "primary" : 仅第一个可用模型
    """
    if not results:
        return {"action": "持有", "confidence": 0.0, "reason": "no_results"}

    names = list(results.keys())
    items = list(results.values())

    if mode == "primary" or len(items) == 1:
        return items[0]

    ranks = [_ACTION_RANK.get(r["action"], 2) for r in items]
    confs = [r["confidence"] for r in items]
    reasons = [r["reason"] for r in items]
    actions = [r["action"] for r in items]

    if mode == "avg":
        avg_conf = sum(confs) / len(confs)
        # 取多数投票的动作
        from collections import Counter
        action_votes = Counter(actions)
        majority_action = action_votes.most_common(1)[0][0]
        label_parts = " | ".join(f"[{_MODEL_LABEL.get(n, n)[:2]}] {reasons[i]}" for i, n in enumerate(names))
        return {
            "action": majority_action,
            "confidence": avg_conf,
            "reason": label_parts,
        }

    # mode == "cross"
    from collections import Counter
    action_votes = Counter(actions)
    most_common_action, most_common_count = action_votes.most_common(1)[0]

    if most_common_count == len(items):
        # 全体一致 → 增强信心
        boosted_conf = min(sum(confs) / len(confs) * 1.15, 1.0)
        return {
            "action": most_common_action,
            "confidence": boosted_conf,
            "reason": f"[{len(items)}模型一致:{most_common_action}] {reasons[0]}",
        }

    if most_common_count >= 2:
        # 多数一致（2/3）→ 取多数意见，但信心打折
        avg_conf = sum(confs) / len(confs)
        minority_names = [n for n, r in zip(names, items) if r["action"] != most_common_action]
        minority_label = ",".join(_MODEL_LABEL.get(n, n)[:2] for n in minority_names)
        return {
            "action": most_common_action,
            "confidence": avg_conf,
            "reason": f"[多数:{most_common_action}，{minority_label}不同意] {reasons[0]}",
        }

    # 三方分歧 → 保守处理
    min_rank = min(ranks)
    conservative_action = _RANK_ACTION.get(min_rank, "持有")
    min_conf = min(confs) * 0.5
    score_parts = " ".join(f"{_MODEL_LABEL.get(n, n)[:2]}:{actions[i]}" for i, n in enumerate(names))
    return {
        "action": conservative_action,
        "confidence": min_conf,
        "reason": f"[三方分歧→保守] {score_parts} | {reasons[0]}",
    }


def batch_portfolio_advice(
    tickers: list[str],
    signal_table: pd.DataFrame,
    fundamentals: pd.DataFrame,
    cfg: LLMConfig,
    sector_desc_map: dict[str, str] | None = None,
    news_ctx: object | None = None,
) -> dict[str, AdviceResult]:
    out: dict[str, AdviceResult] = {}
    sdmap = sector_desc_map or {}

    _format_news: Any = None
    if news_ctx is not None:
        try:
            from .news import format_news_for_prompt
            _format_news = format_news_for_prompt
        except ImportError:
            pass

    if not cfg.enabled:
        for t in tickers:
            out[t] = {"action": "持有", "confidence": 0.0, "reason": "LLM未启用，仅参考量化分数"}
        return out

    models = cfg.available_models
    if not models:
        for t in tickers:
            out[t] = {"action": "持有", "confidence": 0.0, "reason": "无可用模型"}
        return out

    multi = len(models) > 1
    model_names = " + ".join(_MODEL_LABEL.get(m, m) for m in models)

    logger.info("========== 开始生成持仓投资建议 (共%d只, %s%s) ==========",
                len(tickers), model_names,
                f", {cfg.cross_validation_mode}模式" if multi else "")

    import time as _time

    for i, t in enumerate(tickers, 1):
        if t not in signal_table.index:
            out[t] = {"action": "持有", "confidence": 0.0, "reason": "未在评分表中"}
            continue
        row = signal_table.loc[t].to_dict()
        fund_row = fundamentals.loc[t].to_dict() if t in fundamentals.index else {}
        news_text = _format_news(t, news_ctx) if _format_news and news_ctx else ""
        prompt = _build_advice_prompt(t, row, fund_row, sector_desc=sdmap.get(t, ""), news_text=news_text)

        advice_results: dict[str, AdviceResult] = {}
        for m in models:
            label = _MODEL_LABEL.get(m, m)
            try:
                raw = _call_model_raw(cfg, m, prompt)
                if raw:
                    advice_results[m] = _parse_advice_safe(raw)
                    logger.info("建议 [%d/%d] %s %s: %s (信心%.0f%%)",
                                i, len(tickers), t, label,
                                advice_results[m]["action"],
                                advice_results[m]["confidence"] * 100)
            except Exception as e:
                logger.warning("建议 [%d/%d] %s %s异常: %s", i, len(tickers), t, label, e)

        if not advice_results:
            out[t] = {"action": "持有", "confidence": 0.0, "reason": "所有模型调用失败"}
        elif len(advice_results) == 1:
            out[t] = list(advice_results.values())[0]
        else:
            advice = _cross_validate_advice(advice_results, cfg.cross_validation_mode)
            actions_str = " ".join(f"{_MODEL_LABEL.get(m, m)[:2]}={advice_results[m]['action']}" for m in advice_results)
            logger.info("建议 [%d/%d] %s 融合: %s → %s (信心%.0f%%)",
                        i, len(tickers), t, actions_str, advice["action"], advice["confidence"] * 100)
            out[t] = advice

        _time.sleep(0.3)

    logger.info("========== 持仓建议生成完成 ==========")
    return out


def portfolio_overall_analysis(
    signal_table: pd.DataFrame,
    fundamentals: pd.DataFrame,
    cfg: LLMConfig,
    sector_desc_map: dict[str, str] | None = None,
    news_ctx: object | None = None,
) -> str:
    """对持仓组合进行整体分析。多模型时各自独立分析，再用一个模型汇总。"""
    if not cfg.enabled:
        return _portfolio_overall_fallback(signal_table, fundamentals)

    _format_news: Any = None
    if news_ctx is not None:
        try:
            from .news import format_news_for_prompt
            _format_news = format_news_for_prompt
        except ImportError:
            pass

    prompt = _build_portfolio_overall_prompt(
        signal_table, fundamentals, sector_desc_map or {},
        _format_news, news_ctx,
    )
    logger.info("========== 开始生成持仓整体分析 ==========")

    models = cfg.available_models
    if not models:
        return _portfolio_overall_fallback(signal_table, fundamentals)

    # 各模型独立分析
    analyses: dict[str, str] = {}
    for m in models:
        label = _MODEL_LABEL.get(m, m)
        try:
            raw = _call_model_raw(cfg, m, prompt)
            if raw:
                parsed = _parse_portfolio_overall(raw)
                analyses[m] = parsed
                logger.info("整体分析 %s 完成 (%d字)", label, len(parsed))
        except Exception as e:
            logger.warning("整体分析 %s 异常: %s", label, e)

    if not analyses:
        return _portfolio_overall_fallback(signal_table, fundamentals)

    if len(analyses) == 1:
        logger.info("持仓整体分析生成成功（单模型）")
        return list(analyses.values())[0]

    # 多模型汇总：用第一个可用模型将所有分析整合
    analyst_blocks = "\n\n".join(
        f"【分析师{chr(65 + i)}（{_MODEL_LABEL.get(m, m)}）】\n{text}"
        for i, (m, text) in enumerate(analyses.items())
    )
    merge_prompt = (
        "你是资深美股/港股投资组合顾问。下面是多位AI分析师对同一持仓组合的独立分析，"
        "请将所有分析汇总整合为一份最终报告。\n\n"
        "要求：\n"
        "- 综合所有分析师的观点，取长补短\n"
        "- 多数一致的观点着重强调\n"
        "- 存在分歧的地方都列出并标注分歧\n"
        "- 保持五个维度的结构（含总仓位建议）\n"
        "- 总仓位只能从 100%/75%/50%/25%/0% 中选一个\n\n"
        f"{analyst_blocks}\n\n"
        "请输出JSON格式：\n"
        '{"summary":"一句话总评","risk":"风险分析(2-3句)","rebalance":"调仓建议(2-3句)","watchlist":"关注要点(1-2句)","position":"75%","position_reason":"理由(1-2句)"}'
    )

    # 用第一个可用模型做汇总
    merge_model = list(analyses.keys())[0]
    try:
        merged_raw = _call_model_raw(cfg, merge_model, merge_prompt)
        merged_analysis = _parse_portfolio_overall(merged_raw)
        model_names = "+".join(_MODEL_LABEL.get(m, m) for m in analyses)
        logger.info("持仓整体分析生成成功（%s 汇总）", model_names)
        return merged_analysis
    except Exception as e:
        logger.warning("汇总分析异常: %s，返回第一份分析", e)
        return list(analyses.values())[0]


def _build_portfolio_overall_prompt(
    signal_table: pd.DataFrame,
    fundamentals: pd.DataFrame,
    sector_desc_map: dict[str, str],
    format_news_fn: Any,
    news_ctx: object | None,
) -> str:
    """构建持仓整体分析的 prompt。"""
    lines: list[str] = []

    # 汇总每只持仓的关键信息
    for t in signal_table.index:
        row = signal_table.loc[t]
        name = str(row.get("name", t))
        engine = str(row.get("engine", ""))
        final_score = _to_float(row.get("final_score", 0))
        action = str(row.get("action", ""))
        advice_action = str(row.get("advice_action", ""))
        vol = row.get("annual_vol", None)
        vol_str = f"{float(vol):.0%}" if vol is not None and pd.notna(vol) else "N/A"

        fund = fundamentals.loc[t] if t in fundamentals.index else pd.Series()
        # 行业：杠杆ETF优先用行业描述，其次用fundamentals，兜底用引擎类型
        if t in sector_desc_map:
            sector = sector_desc_map[t].split("，")[0]  # 取描述的第一句作为简称
        elif not fund.empty and pd.notna(fund.get("sector")):
            sector = str(fund["sector"])
        elif engine == "leverage":
            sector = "杠杆ETF"
        else:
            sector = "Unknown"
        market_cap = _to_float(fund.get("market_cap", 0))
        rev_growth = _to_float(fund.get("revenue_growth", 0))

        cap_str = f"{market_cap / 1e9:.1f}B" if market_cap > 0 else "N/A"
        lines.append(
            f"  {t} ({name}) | 引擎:{engine} | 行业:{sector} | 市值:{cap_str} | "
            f"营收增速:{rev_growth:.0%} | 波动率:{vol_str} | 得分:{final_score:.1f} | "
            f"信号:{action} | 建议:{advice_action}"
        )

    holdings_block = "\n".join(lines)

    # 行业分布统计
    sector_counts: dict[str, int] = {}
    engine_counts: dict[str, int] = {}
    for t in signal_table.index:
        row = signal_table.loc[t]
        eng = str(row.get("engine", "other"))
        engine_counts[eng] = engine_counts.get(eng, 0) + 1
        # 行业统计：与上面保持一致的逻辑
        if t in sector_desc_map:
            sec = sector_desc_map[t].split("，")[0]
        else:
            fund = fundamentals.loc[t] if t in fundamentals.index else pd.Series()
            if not fund.empty and pd.notna(fund.get("sector")):
                sec = str(fund["sector"])
            elif eng == "leverage":
                sec = "杠杆ETF"
            else:
                sec = "Other"
        sector_counts[sec] = sector_counts.get(sec, 0) + 1

    sector_dist = ", ".join(f"{k}:{v}" for k, v in sorted(sector_counts.items(), key=lambda x: -x[1]))
    engine_dist = ", ".join(f"{k}:{v}" for k, v in sorted(engine_counts.items(), key=lambda x: -x[1]))

    # 新闻摘要
    news_block = ""
    if format_news_fn and news_ctx:
        news_items = []
        for t in signal_table.index[:5]:  # 前5只持仓的新闻
            news_text = format_news_fn(t, news_ctx)
            if news_text:
                news_items.append(news_text)
        if news_items:
            news_block = "\n【近期相关新闻】\n" + "\n".join(news_items) + "\n"

    avg_score = signal_table["final_score"].mean() if "final_score" in signal_table.columns else 0
    avg_vol_raw = signal_table["annual_vol"].dropna().mean() if "annual_vol" in signal_table.columns else None
    avg_vol_str = f"{avg_vol_raw:.0%}" if avg_vol_raw is not None and pd.notna(avg_vol_raw) else "N/A"

    return (
        "你是资深美股/港股投资组合顾问。请对我当前的整体持仓进行全面分析。\n\n"
        f"【持仓概况】共 {len(signal_table)} 只标的\n"
        f"引擎分布: {engine_dist}\n"
        f"行业分布: {sector_dist}\n"
        f"平均得分: {avg_score:.1f}, 平均波动率: {avg_vol_str}\n\n"
        f"【逐只持仓】\n{holdings_block}\n"
        f"{news_block}\n"
        "请从以下维度进行整体分析，用中文回答：\n"
        "1. 📊 组合总评：当前组合整体状态（健康/需调整/风险偏高），一句话总结\n"
        "2. ⚠️ 风险提示：行业集中度风险、杠杆敞口风险、波动率风险、相关性风险\n"
        "3. 💡 调仓建议：最值得加仓/减仓的标的，以及理由\n"
        "4. 🎯 关注要点：近期需要重点关注的宏观/行业事件\n"
        "5. 💰 总仓位建议：基于当前宏观环境、市场风险、VIX水平、地缘政治等因素，"
        "建议总仓位应控制在多少？只能从以下选项中选一个：100%（满仓进攻）、75%（积极偏多）、50%（攻守平衡）、25%（防守为主）、0%（空仓观望）。"
        "并给出1-2句理由。\n\n"
        "请输出JSON格式：\n"
        '{"summary":"一句话总评","risk":"风险分析(2-3句)","rebalance":"调仓建议(2-3句)","watchlist":"关注要点(1-2句)","position":"75%","position_reason":"理由(1-2句)"}'
    )


def _parse_portfolio_overall(raw: str) -> str:
    """解析整体分析的 JSON 并格式化为可读文本。"""
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
        parts = []
        if data.get("summary"):
            parts.append(f"  📊 组合总评: {data['summary']}")
        if data.get("risk"):
            parts.append(f"  ⚠️  风险提示: {data['risk']}")
        if data.get("rebalance"):
            parts.append(f"  💡 调仓建议: {data['rebalance']}")
        if data.get("watchlist"):
            parts.append(f"  🎯 关注要点: {data['watchlist']}")
        if data.get("position"):
            pos_reason = f" — {data['position_reason']}" if data.get("position_reason") else ""
            parts.append(f"  💰 总仓位建议: {data['position']}{pos_reason}")
        return "\n".join(parts) if parts else raw
    except Exception:
        # JSON 解析失败，直接返回原文
        return raw.strip()[:800]


def _portfolio_overall_fallback(signal_table: pd.DataFrame, fundamentals: pd.DataFrame) -> str:
    """LLM 未启用时，基于纯量化数据生成简要分析。"""
    n = len(signal_table)
    if n == 0:
        return "  无持仓数据"

    avg_score = signal_table["final_score"].mean() if "final_score" in signal_table.columns else 0
    avg_vol_raw = signal_table["annual_vol"].dropna().mean() if "annual_vol" in signal_table.columns else None
    avg_vol_str = f"{avg_vol_raw:.0%}" if avg_vol_raw is not None and pd.notna(avg_vol_raw) else "N/A"

    buy_count = (signal_table["action"] == "BUY").sum() if "action" in signal_table.columns else 0
    reduce_count = (signal_table["action"] == "REDUCE").sum() if "action" in signal_table.columns else 0

    # 行业集中度
    sectors: dict[str, int] = {}
    for t in signal_table.index:
        fund = fundamentals.loc[t] if t in fundamentals.index else pd.Series()
        eng = str(signal_table.loc[t].get("engine", "")) if t in signal_table.index else ""
        if not fund.empty and pd.notna(fund.get("sector")):
            sec = str(fund["sector"])
        elif eng == "leverage":
            sec = "杠杆ETF"
        else:
            sec = "Other"
        sectors[sec] = sectors.get(sec, 0) + 1
    top_sector = max(sectors, key=sectors.get) if sectors else "N/A"  # type: ignore[arg-type]
    top_sector_pct = sectors.get(top_sector, 0) / n * 100

    # 杠杆占比
    lev_count = (signal_table["engine"] == "leverage").sum() if "engine" in signal_table.columns else 0
    lev_pct = lev_count / n * 100

    parts = [
        f"  📊 组合总评: 共{n}只持仓, 平均得分{avg_score:.1f}, 平均波动率{avg_vol_str}, BUY信号{buy_count}只, REDUCE信号{reduce_count}只",
        f"  ⚠️  风险提示: 行业集中度—{top_sector}占{top_sector_pct:.0f}%, 杠杆敞口{lev_pct:.0f}%({lev_count}只杠杆ETF)",
        f"  💡 调仓建议: (LLM未启用，请开启--llm获取AI调仓建议)",
    ]
    return "\n".join(parts)



