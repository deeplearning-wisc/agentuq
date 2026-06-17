import json
import os
import re
from typing import Any, Optional

import litellm
from litellm import completion, completion_cost
from litellm.caching.caching import Cache
from litellm.main import ModelResponse, Usage
from loguru import logger

from tau2.config import (
    DEFAULT_LLM_CACHE_TYPE,
    DEFAULT_MAX_RETRIES,
    LLM_CACHE_ENABLED,
    REDIS_CACHE_TTL,
    REDIS_CACHE_VERSION,
    REDIS_HOST,
    REDIS_PASSWORD,
    REDIS_PORT,
    REDIS_PREFIX,
    USE_LANGFUSE,
)
from tau2.data_model.message import (
    AssistantMessage,
    Message,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.environment.tool import Tool
from tau2.utils.uq_logprobs import save_response_logprobs, strip_logprobs_from_response

# litellm._turn_on_debug()

if USE_LANGFUSE:
    # set callbacks
    litellm.success_callback = ["langfuse"]
    litellm.failure_callback = ["langfuse"]

litellm.drop_params = True

if LLM_CACHE_ENABLED:
    if DEFAULT_LLM_CACHE_TYPE == "redis":
        logger.info(f"LiteLLM: Using Redis cache at {REDIS_HOST}:{REDIS_PORT}")
        litellm.cache = Cache(
            type=DEFAULT_LLM_CACHE_TYPE,
            host=REDIS_HOST,
            port=REDIS_PORT,
            password=REDIS_PASSWORD,
            namespace=f"{REDIS_PREFIX}:{REDIS_CACHE_VERSION}:litellm",
            ttl=REDIS_CACHE_TTL,
        )
    elif DEFAULT_LLM_CACHE_TYPE == "local":
        logger.info("LiteLLM: Using local cache")
        litellm.cache = Cache(
            type="local",
            ttl=REDIS_CACHE_TTL,
        )
    else:
        raise ValueError(
            f"Invalid cache type: {DEFAULT_LLM_CACHE_TYPE}. Should be 'redis' or 'local'"
        )
    litellm.enable_cache()
else:
    logger.info("LiteLLM: Cache is disabled")
    litellm.disable_cache()


ALLOW_SONNET_THINKING = False

if not ALLOW_SONNET_THINKING:
    logger.warning("Sonnet thinking is disabled")


def _parse_ft_model_name(model: str) -> str:
    """
    Parse the ft model name from the litellm model name.
    e.g: "ft:gpt-4.1-mini-2025-04-14:sierra::BSQA2TFg" -> "gpt-4.1-mini-2025-04-14"
    """
    pattern = r"ft:(?P<model>[^:]+):(?P<provider>\w+)::(?P<id>\w+)"
    match = re.match(pattern, model)
    if match:
        return match.group("model")
    else:
        return model


def get_response_cost(response: ModelResponse) -> float:
    """
    Get the cost of the response from the litellm completion.
    """
    response.model = _parse_ft_model_name(
        response.model
    )  # FIXME: Check Litellm, passing the model to completion_cost doesn't work.
    try:
        cost = completion_cost(completion_response=response)
    except Exception:
        # logger.error(e)
        return 0.0
    return cost


def get_response_usage(response: ModelResponse) -> Optional[dict]:
    usage: Optional[Usage] = response.get("usage")
    if usage is None:
        return None
    return {
        "completion_tokens": usage.completion_tokens,
        "prompt_tokens": usage.prompt_tokens,
    }


def to_tau2_messages(
    messages: list[dict], ignore_roles: set[str] = set()
) -> list[Message]:
    """
    Convert a list of messages from a dictionary to a list of Tau2 messages.
    """
    tau2_messages = []
    for message in messages:
        role = message["role"]
        if role in ignore_roles:
            continue
        if role == "user":
            tau2_messages.append(UserMessage(**message))
        elif role == "assistant":
            tau2_messages.append(AssistantMessage(**message))
        elif role == "tool":
            tau2_messages.append(ToolMessage(**message))
        elif role == "system":
            tau2_messages.append(SystemMessage(**message))
        else:
            raise ValueError(f"Unknown message type: {role}")
    return tau2_messages


def to_litellm_messages(messages: list[Message]) -> list[dict]:
    """
    Convert a list of Tau2 messages to a list of litellm messages.
    """
    litellm_messages = []
    for message in messages:
        if isinstance(message, UserMessage):
            litellm_messages.append({"role": "user", "content": message.content})
        elif isinstance(message, AssistantMessage):
            tool_calls = None
            if message.is_tool_call():
                tool_calls = [
                    {
                        "id": tc.id,
                        "name": tc.name,
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                        "type": "function",
                    }
                    for tc in message.tool_calls
                ]
            litellm_messages.append(
                {
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": tool_calls,
                }
            )
        elif isinstance(message, ToolMessage):
            litellm_messages.append(
                {
                    "role": "tool",
                    "content": message.content,
                    "tool_call_id": message.id,
                }
            )
        elif isinstance(message, SystemMessage):
            litellm_messages.append({"role": "system", "content": message.content})
    return litellm_messages


def generate(
    model: str,
    messages: list[Message],
    tools: Optional[list[Tool]] = None,
    tool_choice: Optional[str] = None,
    caller_role: str = "assistant",
    **kwargs: Any,
) -> UserMessage | AssistantMessage:
    """
    Generate a response from the model.

    Args:
        model: The model to use.
        messages: The messages to send to the model.
        tools: The tools to use.
        tool_choice: The tool choice to use.
        caller_role: The logical role of the caller for UQ logprobs tagging.
            Use "assistant" for agent calls (default) and "user" for user
            simulator calls.
        **kwargs: Additional arguments to pass to the model.

    Returns: A tuple containing the message and the cost.
    """
    if kwargs.get("num_retries") is None:
        kwargs["num_retries"] = DEFAULT_MAX_RETRIES

    if model.startswith("claude") and not ALLOW_SONNET_THINKING:
        kwargs["thinking"] = {"type": "disabled"}

    max_empty_retries = kwargs.pop("max_empty_retries", 2)
    litellm_messages = to_litellm_messages(messages)
    tools = [tool.openai_schema for tool in tools] if tools else None
    if tools and tool_choice is None:
        tool_choice = "auto"

    def _normalize_content(value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            chunks: list[str] = []
            for item in value:
                if isinstance(item, str):
                    chunks.append(item)
                elif isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        chunks.append(text)
            normalized = "\n".join(c.strip() for c in chunks if c and c.strip())
            return normalized or None
        return str(value)

    def _parse_tool_call_arguments(arguments: Any) -> dict:
        if isinstance(arguments, dict):
            return arguments
        if arguments is None:
            return {}
        if isinstance(arguments, (bytes, bytearray)):
            arguments = arguments.decode("utf-8", errors="replace")
        if isinstance(arguments, str):
            stripped = arguments.strip()
            if stripped == "":
                return {}
            return json.loads(stripped)
        raise TypeError(f"Unsupported tool call arguments type: {type(arguments)}")

    message = None
    full_response_dict = None
    response_choice_dict = None
    cost = None
    usage = None
    for attempt in range(max_empty_retries + 1):
        try:
            response = completion(
                model=model,
                messages=litellm_messages,
                tools=tools,
                tool_choice=tool_choice,
                **kwargs,
            )
        except Exception as e:
            logger.error(e)
            raise e

        full_response_dict = response.to_dict()
        cost = get_response_cost(response)
        usage = get_response_usage(response)
        response = response.choices[0]
        response_choice_dict = response.to_dict()
        try:
            finish_reason = response.finish_reason
            if finish_reason == "length":
                logger.warning("Output might be incomplete due to token limit!")
        except Exception as e:
            logger.error(e)
            raise e
        assert response.message.role == "assistant", (
            "The response should be an assistant message"
        )
        content = _normalize_content(response.message.content)
        tool_calls = response.message.tool_calls or []
        parsed_tool_calls: list[ToolCall] = []
        tool_parse_error = None
        for tool_call in tool_calls:
            raw_arguments = tool_call.function.arguments
            try:
                arguments = _parse_tool_call_arguments(raw_arguments)
            except Exception as e:
                tool_parse_error = e
                raw_preview = str(raw_arguments)
                if len(raw_preview) > 200:
                    raw_preview = raw_preview[:200] + "..."
                if attempt < max_empty_retries:
                    logger.warning(
                        f"Invalid tool call arguments JSON (attempt {attempt + 1}/{max_empty_retries + 1}) for tool `{tool_call.function.name}`: {raw_preview}. Retrying..."
                    )
                    break
                raise ValueError(
                    f"Invalid tool call arguments for tool `{tool_call.function.name}` after {max_empty_retries + 1} attempts: {raw_preview}"
                ) from e
            parsed_tool_calls.append(
                ToolCall(
                    id=tool_call.id,
                    name=tool_call.function.name,
                    arguments=arguments,
                )
            )
        if tool_parse_error is not None and attempt < max_empty_retries:
            continue
        tool_calls = parsed_tool_calls or None

        # Some providers can return an assistant message with no content/tool call
        # (e.g., transient empty choices). Try to recover using refusal text first.
        if content is None and tool_calls is None:
            refusal = _normalize_content(getattr(response.message, "refusal", None))
            if refusal is not None:
                content = refusal

        message = AssistantMessage(
            role="assistant",
            content=content,
            tool_calls=tool_calls,
            cost=cost,
            usage=usage,
            raw_data=strip_logprobs_from_response(response_choice_dict),
        )
        if message.has_text_content() or message.is_tool_call():
            break
        if attempt < max_empty_retries:
            logger.warning(
                f"Model returned empty assistant message (attempt {attempt + 1}/{max_empty_retries + 1}). Retrying..."
            )
        else:
            logger.error(
                f"Model returned empty assistant message after {max_empty_retries + 1} attempts."
            )
            message.content = "I'm sorry, I could not produce a valid response."
    uq_logprobs_info = save_response_logprobs(
        raw_response=full_response_dict,
        model=model,
        role=caller_role,
        message_timestamp=message.timestamp,
        turn_idx=message.turn_idx,
        request_params={
            "temperature": kwargs.get("temperature"),
            "logprobs": kwargs.get("logprobs"),
            "top_logprobs": kwargs.get("top_logprobs"),
        },
    )
    if uq_logprobs_info is not None:
        if message.raw_data is None:
            message.raw_data = {}
        message.raw_data["uq_logprobs"] = uq_logprobs_info
    return message


def get_cost(messages: list[Message]) -> tuple[float, float] | None:
    """
    Get the cost of the interaction between the agent and the user.
    Returns None if any message has no cost.
    """
    agent_cost = 0
    user_cost = 0
    for message in messages:
        if isinstance(message, ToolMessage):
            continue
        if message.cost is not None:
            if isinstance(message, AssistantMessage):
                agent_cost += message.cost
            elif isinstance(message, UserMessage):
                user_cost += message.cost
        else:
            logger.warning(f"Message {message.role}: {message.content} has no cost")
            return None
    return agent_cost, user_cost


def get_token_usage(messages: list[Message]) -> dict:
    """
    Get the token usage of the interaction between the agent and the user.
    """
    usage = {"completion_tokens": 0, "prompt_tokens": 0}
    for message in messages:
        if isinstance(message, ToolMessage):
            continue
        if message.usage is None:
            logger.warning(f"Message {message.role}: {message.content} has no usage")
            continue
        usage["completion_tokens"] += message.usage["completion_tokens"]
        usage["prompt_tokens"] += message.usage["prompt_tokens"]
    return usage
