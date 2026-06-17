"""Shared rescoring utilities for UQ pipelines.

This module provides common functions for teacher-forced rescoring of
messages via OpenAI-compatible completion and chat APIs.  Extracted from
``tau2.scripts.extract_uq_from_trajectories`` so that both the extraction
script and the observation-UQ scorer can reuse the same logic.
"""

import json
import urllib.error
import urllib.request
from typing import Any, Optional


def post_json(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout: int = 120,
) -> dict[str, Any]:
    """Send an HTTP POST with a JSON body and return the parsed response."""
    req = urllib.request.Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body)
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"HTTPError {e.code}: {err_body}") from e


def normalize_openai_model_name(model: str) -> str:
    """Strip provider prefixes (``openai/``, ``azure/``) that litellm uses
    for routing but that raw API endpoints don't understand."""
    for prefix in ("openai/", "azure/"):
        if model.startswith(prefix):
            return model.split(prefix, 1)[1]
    return model


def message_to_target_text(message: dict[str, Any]) -> Optional[str]:
    """Extract the scorable target text from a message dict."""
    content = message.get("content")
    if isinstance(content, str) and content != "":
        return content
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list) and len(tool_calls) > 0:
        return json.dumps(tool_calls, ensure_ascii=False, sort_keys=True)
    return None


def message_to_context_line(message: dict[str, Any]) -> Optional[str]:
    """Format a message dict as a single text line for completion-style prompts."""
    role = message.get("role")
    if role not in {"system", "user", "assistant", "tool"}:
        return None
    content = message.get("content")
    tool_calls = message.get("tool_calls")
    if isinstance(content, str) and content != "":
        return f"{role.upper()}: {content}"
    if isinstance(tool_calls, list) and len(tool_calls) > 0:
        return f"{role.upper()}_TOOL_CALLS: {json.dumps(tool_calls, ensure_ascii=False, sort_keys=True)}"
    return f"{role.upper()}:"


def to_chat_message(message: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Convert a trajectory message dict to an OpenAI chat-API message.

    Handles both the tau2 trajectory format (flat ``name``/``arguments``)
    and the OpenAI chat-API format (nested ``function.name`` /
    ``function.arguments``), so the function is safe to call on messages
    that have already been converted.
    """
    role = message.get("role")
    if role == "system":
        return {"role": "system", "content": message.get("content")}
    if role == "user":
        return {"role": "user", "content": message.get("content")}
    if role == "assistant":
        if (
            isinstance(message.get("tool_calls"), list)
            and len(message.get("tool_calls")) > 0
        ):
            tool_calls = []
            for tc in message["tool_calls"]:
                if not isinstance(tc, dict):
                    continue
                # Support both flat (trajectory) and nested (chat-API) formats.
                func = tc.get("function") or {}
                name = tc.get("name") or func.get("name")
                args = tc.get("arguments") or func.get("arguments")
                # Arguments may already be a JSON string (chat-API format)
                # or a dict (trajectory format).
                if isinstance(args, dict):
                    args_str = json.dumps(args, ensure_ascii=False)
                elif isinstance(args, str):
                    args_str = args
                else:
                    args_str = "{}"
                tool_calls.append(
                    {
                        "id": tc.get("id", ""),
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": args_str,
                        },
                    }
                )
            return {
                "role": "assistant",
                "content": message.get("content"),
                "tool_calls": tool_calls,
            }
        return {"role": "assistant", "content": message.get("content")}
    if role == "tool":
        return {
            "role": "tool",
            "content": message.get("content"),
            "tool_call_id": message.get("tool_call_id") or message.get("id", ""),
        }
    return None


def build_chat_messages_for_rescore(
    messages: list[dict[str, Any]], idx: int
) -> tuple[list[dict[str, Any]], Optional[str]]:
    """Build the chat-API message list and target text for rescoring at *idx*."""
    if idx < 0 or idx >= len(messages):
        return [], None
    target_text = message_to_target_text(messages[idx])
    if target_text is None:
        return [], None
    chat_messages: list[dict[str, Any]] = []
    for m in messages[:idx]:
        if not isinstance(m, dict):
            continue
        chat_m = to_chat_message(m)
        if chat_m is not None:
            chat_messages.append(chat_m)
    return chat_messages, target_text


def build_prompt_for_rescoring(
    messages: list[dict[str, Any]], idx: int
) -> tuple[Optional[str], Optional[str]]:
    """Build a completion-style prompt with explicit target marker for
    teacher-forced scoring."""
    if idx < 0 or idx >= len(messages):
        return None, None
    target_text = message_to_target_text(messages[idx])
    if target_text is None:
        return None, None
    context_lines = []
    for m in messages[:idx]:
        if not isinstance(m, dict):
            continue
        line = message_to_context_line(m)
        if line is not None:
            context_lines.append(line)
    prompt_prefix = "\n".join(context_lines)
    if prompt_prefix != "":
        prompt_prefix += "\n"
    prompt_prefix += "TARGET_OUTPUT:\n"
    prompt = prompt_prefix + target_text
    return prompt, target_text


def completion_logprobs_to_chat_content(
    completion_response: dict[str, Any],
    *,
    target_start_char: int,
) -> list[dict[str, Any]]:
    """Convert completion-API logprobs to the chat-API ``content`` list format."""
    choices = completion_response.get("choices", [])
    if not isinstance(choices, list) or len(choices) == 0:
        return []
    choice0 = choices[0] if isinstance(choices[0], dict) else {}
    logprobs = choice0.get("logprobs", {})
    if not isinstance(logprobs, dict):
        return []
    tokens = logprobs.get("tokens", [])
    token_logprobs = logprobs.get("token_logprobs", [])
    text_offsets = logprobs.get("text_offset", [])
    top_logprobs = logprobs.get("top_logprobs", [])
    if not (
        isinstance(tokens, list)
        and isinstance(token_logprobs, list)
        and isinstance(text_offsets, list)
    ):
        return []

    n = min(len(tokens), len(token_logprobs), len(text_offsets))
    items: list[dict[str, Any]] = []
    for i in range(n):
        token = tokens[i]
        offset = text_offsets[i]
        if not isinstance(offset, int):
            continue
        if offset < target_start_char:
            continue
        lp = token_logprobs[i]
        topk_dict = top_logprobs[i] if i < len(top_logprobs) else None
        topk_entries = []
        if isinstance(topk_dict, dict):
            for cand_token, cand_lp in topk_dict.items():
                try:
                    topk_entries.append(
                        {"token": str(cand_token), "logprob": float(cand_lp)}
                    )
                except (TypeError, ValueError):
                    continue
        try:
            token_lp = float(lp)
        except (TypeError, ValueError):
            token_lp = None
        if token_lp is None:
            continue
        items.append(
            {
                "token": token,
                "logprob": token_lp,
                "top_logprobs": topk_entries,
            }
        )
    return items


def _build_url_and_headers(
    api_base: str,
    api_key: Optional[str],
    api_version: Optional[str],
    endpoint: str,
    model: Optional[str] = None,
) -> tuple[str, dict[str, str]]:
    """Build the request URL and auth headers.

    For Azure OpenAI (when *api_version* is provided), the URL uses the
    ``/openai/deployments/{model}`` path prefix, gets an ``api-version``
    query parameter, and authentication uses the ``api-key`` header.
    Otherwise, the standard ``Authorization: Bearer`` header is used.
    """
    base = api_base.rstrip("/")
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_version:
        # Azure OpenAI: /openai/deployments/{model}{endpoint}?api-version=...
        if model:
            deployment = normalize_openai_model_name(model)
            url = f"{base}/openai/deployments/{deployment}{endpoint}"
        else:
            url = base + endpoint
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}api-version={api_version}"
        if api_key:
            headers["api-key"] = api_key
    else:
        url = base + endpoint
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
    return url, headers


def rescore_message_with_completions(
    *,
    messages: list[dict[str, Any]],
    idx: int,
    model: str,
    api_base: str,
    api_key: Optional[str],
    temperature: Optional[float],
    top_k: int,
    timeout_sec: int = 120,
    api_version: Optional[str] = None,
) -> dict[str, Any]:
    """Teacher-forced scoring via ``/v1/completions`` with echo.

    Returns a pseudo chat-style response dict with a ``logprobs.content``
    key, or an empty dict on failure.
    """
    prompt, target_text = build_prompt_for_rescoring(messages=messages, idx=idx)
    if prompt is None or target_text is None:
        return {}
    target_start_char = len(prompt) - len(target_text)
    payload = {
        "model": normalize_openai_model_name(model),
        "prompt": prompt,
        "max_tokens": 0,
        "temperature": 0.0 if temperature is None else temperature,
        "logprobs": int(top_k),
        "echo": True,
    }
    url, headers = _build_url_and_headers(
        api_base, api_key, api_version, "/completions", model=model
    )
    response = post_json(
        url=url,
        payload=payload,
        headers=headers,
        timeout=timeout_sec,
    )
    content_items = completion_logprobs_to_chat_content(
        completion_response=response,
        target_start_char=target_start_char,
    )
    if len(content_items) == 0:
        return {}
    pseudo_chat_response = {"logprobs": {"content": content_items}}
    return pseudo_chat_response


def rescore_message_with_chat_replay(
    *,
    messages: list[dict[str, Any]],
    idx: int,
    model: str,
    api_base: str,
    api_key: Optional[str],
    temperature: Optional[float],
    top_k: int,
    timeout_sec: int = 120,
    require_exact_match: bool = True,
    api_version: Optional[str] = None,
) -> tuple[dict[str, Any], bool]:
    """Scoring via ``/v1/chat/completions`` with chat replay.

    Returns ``(pseudo_response, exact_match)`` where *pseudo_response* is
    a dict shaped like a single-choice OpenAI response, or ``{}`` on
    failure.
    """
    chat_messages, target_text = build_chat_messages_for_rescore(
        messages=messages, idx=idx
    )
    if len(chat_messages) == 0 or target_text is None:
        return {}, False

    max_tokens = max(16, min(2048, int(len(target_text) / 2) + 16))
    payload = {
        "model": normalize_openai_model_name(model),
        "messages": chat_messages,
        "max_tokens": max_tokens,
        "temperature": 0.0 if temperature is None else temperature,
        "logprobs": True,
        "top_logprobs": int(top_k),
    }
    url, headers = _build_url_and_headers(
        api_base, api_key, api_version, "/chat/completions", model=model
    )
    response = post_json(
        url=url,
        payload=payload,
        headers=headers,
        timeout=timeout_sec,
    )
    choices = response.get("choices", [])
    if not isinstance(choices, list) or len(choices) == 0:
        return {}, False
    choice0 = choices[0] if isinstance(choices[0], dict) else {}
    message = choice0.get("message", {})
    generated_text = ""
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            generated_text = content
    exact_match = generated_text == target_text
    if require_exact_match and not exact_match:
        return {}, False
    pseudo = {"choices": [choice0]}
    return pseudo, exact_match
