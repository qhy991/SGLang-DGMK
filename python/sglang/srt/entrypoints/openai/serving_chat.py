from __future__ import annotations

import copy
import json
import logging
import time
from http import HTTPStatus
from typing import TYPE_CHECKING, Any, AsyncGenerator, Dict, List, Optional, Union

import jinja2
import orjson
from fastapi import Request
from fastapi.responses import ORJSONResponse, StreamingResponse
from jsonschema import Draft202012Validator, SchemaError

from sglang.srt.entrypoints.openai.encoding_dsv32 import encode_messages
from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    ChatCompletionTokenLogprob,
    ChatMessage,
    ChoiceLogprobs,
    DeltaMessage,
    ErrorResponse,
    FunctionResponse,
    LogProbs,
    MessageProcessingResult,
    SglExt,
    ToolCall,
    ToolCallProcessingResult,
    ToolChoice,
    TopLogprob,
)
from sglang.srt.entrypoints.openai.serving_base import OpenAIServingBase
from sglang.srt.entrypoints.openai.usage_processor import UsageProcessor
from sglang.srt.entrypoints.openai.utils import (
    process_cached_tokens_details_from_ret,
    process_hidden_states_from_ret,
    process_routed_experts_from_ret,
    should_include_usage,
    to_openai_style_logprobs,
)
from sglang.srt.function_call.core_types import ToolCallItem
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.function_call.json_array_parser import JsonArrayParser
from sglang.srt.infini.tool_call_processing import (
    StreamToolCallCollector,
    validate_parsed_tool_call_items,
    validate_required_tool_call_payload,
)
from sglang.srt.infini.fc_token_guard import prefer_engine_finish_on_fc_leak
from sglang.srt.infini.tool_call_validation import ToolCallValidationError
from sglang.srt.infini.kimi_fc_openai_serving import (
    format_openai_tool_call_id,
    history_tool_calls_count,
    is_kimi_k2_openai_serving,
    maybe_strip_kimi_fc_substrings,
    maybe_strip_remaining_tool_args,
    strip_kimi_openai_choice_fields,
)
from sglang.srt.infini.kimi_openai_tool_stream import iter_tool_call_stream_sse_chunks
from sglang.srt.infini.openai_chat_stream_helpers import (
    STREAM_TOOL_DEBUG,
    short_repr,
    sse_stream_plain_text_line,
)
from sglang.srt.function_call.utils import get_json_schema_constraint
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.parser.conversation import generate_chat_conv
from sglang.srt.parser.jinja_template_utils import process_content_for_template_format
from sglang.srt.parser.reasoning_parser import ReasoningParser

if TYPE_CHECKING:
    from sglang.srt.managers.template_manager import TemplateManager
    from sglang.srt.managers.tokenizer_manager import TokenizerManager

logger = logging.getLogger(__name__)


def _tool_parameters_to_json_argument_str(parameters: Any) -> str:
    """Build OpenAI ``function.arguments`` from a decoded ``parameters`` field.

    In ``tool_choice: required`` mode, ``orjson.loads`` may yield ``parameters`` as
    either an object (``dict``) or, if the model nested JSON in a string, a ``str``.
    Applying :func:`json.dumps` to an existing JSON *string* double-encodes it and
    corrupts ``arguments`` for clients.
    """
    if parameters is None:
        return ""
    if isinstance(parameters, str):
        return parameters
    return json.dumps(parameters, ensure_ascii=False)


def _extract_max_dynamic_patch(request: ChatCompletionRequest):
    img_vals = []
    vid_vals = []
    for msg in request.messages or []:
        content = getattr(msg, "content", None)
        if not isinstance(content, list):
            continue
        for part in content:
            # pydantic object or dict type
            if getattr(part, "type", None) == "image_url":
                iu = getattr(part, "image_url", None)
                mdp = getattr(iu, "max_dynamic_patch", None) if iu else None
                if mdp is not None:
                    img_vals.append(int(mdp))
            elif getattr(part, "type", None) == "video_url":
                vu = getattr(part, "video_url", None)
                mdp = getattr(vu, "max_dynamic_patch", None) if vu else None
                if mdp is not None:
                    vid_vals.append(int(mdp))

    # TODO(yuan-luo): per-item max_dynamic_patch for both image and video
    img_max_dynamic_patch = min(img_vals) if img_vals else None
    vid_max_dynamic_patch = min(vid_vals) if vid_vals else None
    return img_max_dynamic_patch, vid_max_dynamic_patch


class OpenAIServingChat(OpenAIServingBase):
    """Handler for /v1/chat/completions requests"""

    _default_sampling_params_logged = False

    def __init__(
        self,
        tokenizer_manager: TokenizerManager,
        template_manager: TemplateManager,
    ):
        super().__init__(tokenizer_manager)
        self.template_manager = template_manager
        self.tool_call_parser = self.tokenizer_manager.server_args.tool_call_parser
        self.reasoning_parser = self.tokenizer_manager.server_args.reasoning_parser

        # Get default sampling parameters from model's generation config
        self.default_sampling_params = (
            self.tokenizer_manager.model_config.get_default_sampling_params()
        )
        if (
            self.default_sampling_params
            and not OpenAIServingChat._default_sampling_params_logged
        ):
            logger.info(
                f"Using default chat sampling params from model generation config: {self.default_sampling_params}",
            )
            OpenAIServingChat._default_sampling_params_logged = True

        # Check if the model is a GPT-OSS model
        self.is_gpt_oss = (
            hasattr(self.tokenizer_manager.model_config, "hf_config")
            and hasattr(self.tokenizer_manager.model_config.hf_config, "model_type")
            and self.tokenizer_manager.model_config.hf_config.model_type == "gpt_oss"
        )

        self.use_dpsk_v32_encoding = self._use_dpsk_v32_encoding()

    def _handle_last_assistant_message(
        self,
        messages: List[Dict[str, Any]],
        request: ChatCompletionRequest,
    ) -> tuple[List[Dict[str, Any]], Optional[str]]:
        """
        Handle continue_final_message feature: separate final assistant message.

        If continue_final_message is enabled and the last message is from assistant,
        extract its content and remove it from the message list.
        If continue_final_message is False and the last message is from assistant,
        convert it to a user message to ensure the last message is always from user.

        Only processes text-based content (strings), ignoring multimodal content (lists).

        Args:
            messages: List of message dictionaries
            request: ChatCompletionRequest with continue_final_message flag

        Returns:
            Tuple of (processed_messages, assistant_prefix)
            - processed_messages: Messages with last assistant message handled appropriately
            - assistant_prefix: Content of the last assistant message (string only), or None
        """
        assistant_prefix = None
        if messages and messages[-1].get("role") == "assistant":
            last_content = messages[-1].get("content")
            # Only process string content, ignore multimodal content (lists)
            if isinstance(last_content, str):
                if request.continue_final_message:
                    # Extract content and remove the assistant message
                    assistant_prefix = last_content
                    messages = messages[:-1]
                else:
                    # Convert the last assistant message to user message
                    messages[-1] = {"role": "user", "content": last_content}
        return messages, assistant_prefix

    def _append_assistant_prefix_to_prompt_ids(
        self, prompt_ids: List[int], assistant_prefix: str
    ) -> List[int]:
        """
        Append assistant prefix to prompt_ids.

        Args:
            prompt_ids: Current prompt token IDs
            assistant_prefix: Assistant message content to append

        Returns:
            Updated prompt_ids with assistant prefix appended
        """
        encoded = self.tokenizer_manager.tokenizer.encode(assistant_prefix)
        if encoded and encoded[0] == self.tokenizer_manager.tokenizer.bos_token_id:
            encoded = encoded[1:]
        return prompt_ids + encoded

    def _use_dpsk_v32_encoding(self) -> bool:
        has_chat_template = (
            self.tokenizer_manager.tokenizer is not None
            and self.tokenizer_manager.tokenizer.chat_template is not None
        )
        architectures = self.tokenizer_manager.model_config.hf_config.architectures
        is_dpsk_v32 = "DeepseekV3" in architectures[0] if architectures else False
        return not has_chat_template and is_dpsk_v32

    def _request_id_prefix(self) -> str:
        return "chatcmpl-"

    def _is_generated_tool_call_validation_enabled(self) -> bool:
        # Gradual rollout: enable runtime tool-call schema validation for kimi_k2 only.
        ret = self.tool_call_parser == "kimi_k2"
        return ret

    def _validate_request(self, request: ChatCompletionRequest) -> Optional[str]:
        """Validate that the input is valid."""
        if not request.messages:
            return "Messages cannot be empty."

        if (
            isinstance(request.tool_choice, str)
            and request.tool_choice.lower() == "required"
            and not request.tools
        ):
            return "Tools cannot be empty if tool choice is set to required."

        if request.tool_choice is not None and not isinstance(request.tool_choice, str):
            if not request.tools:
                return "Tools cannot be empty if tool choice is set to a specific tool."
            tool_name = request.tool_choice.function.name
            tool_exists = any(tool.function.name == tool_name for tool in request.tools)
            if not tool_exists:
                return f"Tool '{tool_name}' not found in tools list."

        # Validate tool definitions
        for i, tool in enumerate(request.tools or []):
            if tool.function.parameters is None:
                continue
            try:
                Draft202012Validator.check_schema(tool.function.parameters)
            except SchemaError as e:
                return f"Tool {i} function has invalid 'parameters' schema: {str(e)}"

        max_output_tokens = request.max_completion_tokens or request.max_tokens
        server_context_length = self.tokenizer_manager.server_args.context_length
        if (
            max_output_tokens
            and server_context_length
            and max_output_tokens > server_context_length
        ) and not self.tokenizer_manager.server_args.allow_auto_truncate:
            return (
                f"max_completion_tokens is too large: {max_output_tokens}."
                f"This model supports at most {server_context_length} completion tokens."
            )

        if request.response_format and request.response_format.type == "json_schema":
            schema = getattr(request.response_format.json_schema, "schema_", None)
            if schema is None:
                return "schema_ is required for json_schema response format request."

        return None

    def _convert_to_internal_request(
        self,
        request: ChatCompletionRequest,
        raw_request: Request = None,
    ) -> tuple[GenerateReqInput, ChatCompletionRequest]:
        reasoning_effort = (
            request.chat_template_kwargs.pop("reasoning_effort", None)
            if request.chat_template_kwargs
            else None
        )
        if self.is_gpt_oss and reasoning_effort == "none":
            raise ValueError(
                f"Harmony does not support reasoning effort {reasoning_effort}"
            )

        if reasoning_effort is not None:
            request.reasoning_effort = reasoning_effort

        """Convert OpenAI chat completion request to internal format"""
        is_multimodal = self.tokenizer_manager.model_config.is_multimodal

        # Compute once and pass through all downstream reasoning consumers.
        require_reasoning = self._get_reasoning_from_request(request)

        # Process messages and apply chat template
        processed_messages = self._process_messages(
            request, is_multimodal, require_reasoning
        )

        # Build sampling parameters
        sampling_params = request.to_sampling_params(
            stop=processed_messages.stop,
            model_generation_config=self.default_sampling_params,
            tool_call_constraint=processed_messages.tool_call_constraint,
        )

        # Handle single vs multiple requests
        if is_multimodal:
            prompt_kwargs = {"text": processed_messages.prompt}
        else:
            if isinstance(processed_messages.prompt_ids, str):
                prompt_kwargs = {"text": processed_messages.prompt_ids}
            else:
                prompt_kwargs = {"input_ids": processed_messages.prompt_ids}

        # Extract custom labels from raw request headers
        custom_labels = self.extract_custom_labels(raw_request)

        # Extract routed_dp_rank from header (has higher priority than body)
        effective_routed_dp_rank = self.extract_routed_dp_rank_from_header(
            raw_request, request.routed_dp_rank
        )

        # Resolve LoRA adapter from model parameter or explicit lora_path
        lora_path = self._resolve_lora_path(request.model, request.lora_path)
        img_max_dynamic_patch, vid_max_dynamic_patch = _extract_max_dynamic_patch(
            request
        )
        adapted_request = GenerateReqInput(
            **prompt_kwargs,
            image_data=processed_messages.image_data,
            video_data=processed_messages.video_data,
            audio_data=processed_messages.audio_data,
            sampling_params=sampling_params,
            return_logprob=request.logprobs,
            logprob_start_len=-1,
            top_logprobs_num=request.top_logprobs or 0,
            stream=request.stream,
            return_text_in_logprobs=True,
            modalities=processed_messages.modalities,
            lora_path=lora_path,
            bootstrap_host=request.bootstrap_host,
            bootstrap_port=request.bootstrap_port,
            bootstrap_room=request.bootstrap_room,
            routed_dp_rank=effective_routed_dp_rank,
            disagg_prefill_dp_rank=request.disagg_prefill_dp_rank,
            return_hidden_states=request.return_hidden_states,
            return_routed_experts=request.return_routed_experts,
            rid=request.rid,
            extra_key=self._compute_extra_key(request),
            require_reasoning=require_reasoning,
            priority=request.priority,
            routing_key=self.extract_routing_key(raw_request),
            custom_labels=custom_labels,
            custom_logit_processor=request.custom_logit_processor,
            image_max_dynamic_patch=img_max_dynamic_patch,
            video_max_dynamic_patch=vid_max_dynamic_patch,
            max_dynamic_patch=getattr(request, "max_dynamic_patch", None),
        )

        return adapted_request, request

    def _process_messages(
        self,
        request: ChatCompletionRequest,
        is_multimodal: bool,
        require_reasoning: bool,
    ) -> MessageProcessingResult:
        """Process chat messages and apply chat template"""
        # GptOss model needs to keep special tokens for harmony parsing
        if self.is_gpt_oss:
            request.skip_special_tokens = False

        self._patch_mistral_skip_special_tokens(request)

        tool_call_constraint = None

        # Apply chat template and its stop strings
        tools = None
        if request.tools and request.tool_choice != "none":
            request.skip_special_tokens = False
            if not isinstance(request.tool_choice, str):
                tools = [
                    item.model_dump()
                    for item in request.tools
                    if item.function.name == request.tool_choice.function.name
                ]
            else:
                tools = [item.model_dump() for item in request.tools]
            if self.tool_call_parser:
                parser = FunctionCallParser(request.tools, self.tool_call_parser)
                tool_call_constraint = parser.get_structure_constraint(
                    request.tool_choice,
                    parallel_tool_calls=request.parallel_tool_calls,
                )
            # Handle JSON schema constraint directly for required or named tool choice
            if request.tool_choice == "required" or isinstance(
                request.tool_choice, ToolChoice
            ):
                json_schema = get_json_schema_constraint(
                    request.tools,
                    request.tool_choice,
                    parallel_tool_calls=request.parallel_tool_calls,
                )
                tool_call_constraint = ("json_schema", json_schema)
        else:
            # construct a structural tag constraint with empty tools;
            tools = []
            tool_call_constraint = ("structural_tag", FunctionCallParser.get_empty_structural_tag())

        # Use chat template
        if self.template_manager.chat_template_name is None:
            result = self._apply_jinja_template(
                request, tools, is_multimodal, require_reasoning
            )
        else:
            result = self._apply_conversation_template(
                request, is_multimodal, require_reasoning
            )

        result.tool_call_constraint = tool_call_constraint
        return result

    def _apply_jinja_template(
        self,
        request: ChatCompletionRequest,
        tools: Optional[List[Dict]],
        is_multimodal: bool,
        require_reasoning: bool,
    ) -> MessageProcessingResult:
        """Apply Jinja chat template"""
        prompt = ""
        prompt_ids = []
        openai_compatible_messages = []
        image_data = []
        video_data = []
        audio_data = []
        modalities = []

        template_content_format = self.template_manager.jinja_template_content_format

        if self.use_dpsk_v32_encoding:
            thinking_mode = "thinking" if require_reasoning else "chat"
            messages = request.messages
            messages = [msg.model_dump() for msg in messages]

            for msg in messages:
                if msg.get("content") is None:
                    msg["content"] = ""
                processed_msg = process_content_for_template_format(
                    msg,
                    template_content_format,
                    image_data,
                    video_data,
                    audio_data,
                    modalities,
                    use_dpsk_v32_encoding=self.use_dpsk_v32_encoding,
                )
                msg.update(processed_msg)

            # Handle continue_final_message: separate final assistant message
            messages, assistant_prefix = self._handle_last_assistant_message(
                messages, request
            )

            if messages[0]["role"] != "system":
                # insert an empty system prompt to help render tool system prompt
                messages.insert(0, {"role": "system", "content": ""})
            if request.tools:
                messages[0]["tools"] = [tool.model_dump() for tool in request.tools]
            real_input = encode_messages(messages, thinking_mode=thinking_mode)
            prompt_ids = self.tokenizer_manager.tokenizer.encode(real_input)

            # Append assistant prefix if continue_final_message is enabled
            if assistant_prefix:
                prompt_ids = self._append_assistant_prefix_to_prompt_ids(
                    prompt_ids, assistant_prefix
                )
        else:
            for message in request.messages:
                if message.content is None:
                    message.content = ""
                msg_dict = message.model_dump()

                # Process content based on detected template format
                processed_msg = process_content_for_template_format(
                    msg_dict,
                    template_content_format,
                    image_data,
                    video_data,
                    audio_data,
                    modalities,
                )

                # per the Transformers docs & maintainers, tool call arguments in
                # assistant-role messages with tool_calls need to be dicts not JSON str -
                # this is how tool-use chat templates will expect them moving forwards
                # so, for messages that have tool_calls, parse the string (which we get
                # from openAI format) to dict
                if (
                    processed_msg["role"] == "assistant"
                    and "tool_calls" in processed_msg
                    and isinstance(processed_msg["tool_calls"], list)
                ):
                    for item in processed_msg["tool_calls"]:
                        if "arguments" in item["function"] and isinstance(
                            item["function"]["arguments"], str
                        ):
                            item["function"]["arguments"] = orjson.loads(
                                item["function"]["arguments"]
                            )

                openai_compatible_messages.append(processed_msg)

            # Handle continue_final_message: separate final assistant message
            openai_compatible_messages, assistant_prefix = (
                self._handle_last_assistant_message(openai_compatible_messages, request)
            )

            extra_template_kwargs = {}
            if request.reasoning_effort is not None:
                extra_template_kwargs["reasoning_effort"] = request.reasoning_effort
            if request.chat_template_kwargs:
                extra_template_kwargs.update(request.chat_template_kwargs)
            self._inject_reasoning_chat_template_kwarg(
                extra_template_kwargs, require_reasoning
            )

            try:
                prompt_ids = self.tokenizer_manager.tokenizer.apply_chat_template(
                    openai_compatible_messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    tools=tools,
                    return_dict=False,
                    **extra_template_kwargs,
                )
            except Exception as e:
                # If the first attempt fails, try with flat function-only format.
                # Some templates (e.g. Mistral) expect tools without the OpenAI wrapper.
                tools = (
                    [t["function"] if "function" in t else t for t in tools]
                    if tools
                    else None
                )
                try:
                    prompt_ids = self.tokenizer_manager.tokenizer.apply_chat_template(
                        openai_compatible_messages,
                        tokenize=True,
                        add_generation_prompt=True,
                        tools=tools,
                        return_dict=False,
                        **extra_template_kwargs,
                    )
                except jinja2.TemplateError as template_error:
                    # Template errors (e.g., from raise_exception in Jinja templates)
                    # should be treated as client errors (400 BadRequest)
                    raise ValueError(str(template_error)) from template_error

            # Append assistant prefix if continue_final_message is enabled
            if assistant_prefix:
                prompt_ids = self._append_assistant_prefix_to_prompt_ids(
                    prompt_ids, assistant_prefix
                )

            if is_multimodal:
                prompt = self.tokenizer_manager.tokenizer.decode(prompt_ids)

        stop = request.stop
        image_data = image_data if image_data else None
        audio_data = audio_data if audio_data else None
        video_data = video_data if video_data else None
        modalities = modalities if modalities else []
        return MessageProcessingResult(
            prompt=prompt,
            prompt_ids=prompt_ids,
            image_data=image_data,
            video_data=video_data,
            audio_data=audio_data,
            modalities=modalities,
            stop=stop,
        )

    def _apply_conversation_template(
        self,
        request: ChatCompletionRequest,
        is_multimodal: bool,
        require_reasoning: bool,
    ) -> MessageProcessingResult:
        """Apply conversation template"""
        prompt = ""
        prompt_ids = []
        conv = generate_chat_conv(request, self.template_manager.chat_template_name)

        # If we should continue the final assistant message, adjust the conversation.
        if (
            request.continue_final_message
            and request.messages
            and request.messages[-1].role == "assistant"
        ):
            # Remove the auto-added blank assistant turn, if present.
            if conv.messages and conv.messages[-1][1] is None:
                conv.messages.pop()
            # Rebuild the prompt from the conversation.
            prompt = conv.get_prompt()
            # Strip trailing stop tokens or separators that indicate end-of-assistant.
            if isinstance(conv.stop_str, list):
                for stop_token in conv.stop_str:
                    if prompt.endswith(stop_token):
                        prompt = prompt[: -len(stop_token)]
            elif isinstance(conv.stop_str, str) and prompt.endswith(conv.stop_str):
                prompt = prompt[: -len(conv.stop_str)]
            if conv.sep and prompt.endswith(conv.sep):
                prompt = prompt[: -len(conv.sep)]
            if getattr(conv, "sep2", None) and prompt.endswith(conv.sep2):
                prompt = prompt[: -len(conv.sep2)]
        else:
            prompt = conv.get_prompt()
            if require_reasoning and self.reasoning_parser not in [
                "qwen3",
                "qwen3-thinking",
                "glm4",
            ]:
                # qwen3 and glm4 think internally without a leading <think> token
                prompt += "<think>"  # Note(Xinyuan): hard code thinking token

        image_data = conv.image_data if conv.image_data else None
        video_data = conv.video_data if conv.video_data else None
        audio_data = conv.audio_data if conv.audio_data else None
        modalities = conv.modalities if conv.modalities else []
        stop = copy.copy(conv.stop_str or [] if not request.ignore_eos else [])

        if request.stop:
            if isinstance(request.stop, str):
                stop.append(request.stop)
            else:
                stop.extend(request.stop)

        if not is_multimodal:
            prompt_ids = self.tokenizer_manager.tokenizer.encode(prompt)

        return MessageProcessingResult(
            prompt=prompt,
            prompt_ids=prompt_ids,
            image_data=image_data,
            video_data=video_data,
            audio_data=audio_data,
            modalities=modalities,
            stop=stop,
        )

    async def _handle_streaming_request(
        self,
        adapted_request: GenerateReqInput,
        request: ChatCompletionRequest,
        raw_request: Request,
    ) -> Union[StreamingResponse, ErrorResponse]:
        """Handle streaming chat completion request"""
        generator = self._generate_chat_stream(adapted_request, request, raw_request)

        # Kick-start the generator to trigger validation before HTTP 200 is sent.
        # If validation fails (e.g., context length exceeded), we can still return
        # a proper HTTP 400 error response instead of streaming it as SSE payload.
        try:
            first_chunk = await generator.__anext__()
        except ValueError as e:
            return self.create_error_response(str(e))

        async def prepend_first_chunk():
            yield first_chunk
            async for chunk in generator:
                yield chunk

        return StreamingResponse(
            prepend_first_chunk(),
            media_type="text/event-stream",
            background=self.tokenizer_manager.create_abort_task(adapted_request),
        )

    async def _generate_chat_stream(
        self,
        adapted_request: GenerateReqInput,
        request: ChatCompletionRequest,
        raw_request: Request,
    ) -> AsyncGenerator[str, None]:
        """Generate streaming chat completion response"""
        # Parsers for tool calls and reasoning
        parser_dict = {}
        reasoning_parser_dict = {}
        stream_tool_call_collector = (
            StreamToolCallCollector(request.tools or [])
            if (
                request.tool_choice != "none"
                and self._is_generated_tool_call_validation_enabled()
            )
            else None
        )

        # State tracking for streaming
        is_firsts = {}
        stream_buffers = {}
        n_prev_tokens = {}
        has_tool_calls = {}
        finish_reasons = {}
        is_kimi = is_kimi_k2_openai_serving(self.tool_call_parser)

        # Usage tracking
        prompt_tokens = {}
        reasoning_tokens = {}
        completion_tokens = {}
        cached_tokens = {}
        hidden_states = {}
        routed_experts = {}

        stream_started = False
        try:
            include_usage, continuous_usage_stats = should_include_usage(
                request.stream_options,
                self.tokenizer_manager.server_args.stream_response_default_include_usage,
            )

            async for content in self.tokenizer_manager.generate_request(
                adapted_request, raw_request
            ):
                index = content.get("index", 0)
                prompt_tokens[index] = content["meta_info"].get("prompt_tokens", 0)
                completion_tokens[index] = content["meta_info"].get(
                    "completion_tokens", 0
                )
                reasoning_tokens[index] = content["meta_info"].get(
                    "reasoning_tokens", 0
                )
                cached_tokens[index] = content["meta_info"].get("cached_tokens", 0)
                hidden_states[index] = content["meta_info"].get("hidden_states", None)
                routed_experts[index] = content["meta_info"].get("routed_experts", None)

                # Handle logprobs
                choice_logprobs = None
                if request.logprobs:
                    n_prev_token = n_prev_tokens.get(index, 0)
                    total_output_logprobs = content["meta_info"][
                        "output_token_logprobs_length"
                    ]
                    if n_prev_token < total_output_logprobs:
                        choice_logprobs = self._process_streaming_logprobs(
                            content, n_prev_token, total_output_logprobs
                        )
                    n_prev_tokens[index] = total_output_logprobs

                finish_reason = content["meta_info"].get("finish_reason", None)
                finish_reason_type = finish_reason["type"] if finish_reason else None

                # Track finish_reason for each index
                if finish_reason_type:
                    # If the abort is from scheduler.
                    if finish_reason_type == "abort":
                        code = finish_reason.get(
                            "status_code", HTTPStatus.INTERNAL_SERVER_ERROR
                        )
                        error = self.create_streaming_error_response(
                            finish_reason.get("message", "Generation aborted."),
                            code.name,
                            code.value,
                        )
                        yield f"data: {error}\n\n"
                        break
                    else:
                        finish_reasons[index] = finish_reason

                # First chunk with role
                if is_firsts.get(index, True):
                    is_firsts[index] = False
                    delta = DeltaMessage(role="assistant")
                    choice_data = ChatCompletionResponseStreamChoice(
                        index=index,
                        delta=delta,
                        finish_reason=None,
                        logprobs=None,
                    )
                    chunk = ChatCompletionStreamResponse(
                        id=content["meta_info"]["id"],
                        created=int(time.time()),
                        choices=[choice_data],
                        model=request.model,
                    )
                    yield f"data: {chunk.model_dump_json()}\n\n"
                    stream_started = True

                stream_buffer = stream_buffers.get(index, "")
                delta = content["text"][len(stream_buffer) :]
                engine_text_delta_len = len(delta)
                stream_buffers[index] = stream_buffer + delta

                # Handle reasoning content
                reasoning_text: Optional[str] = None
                if self.reasoning_parser and request.separate_reasoning:
                    reasoning_text, delta = self._process_reasoning_stream(
                        index,
                        delta,
                        reasoning_parser_dict,
                        content,
                        request,
                        adapted_request.require_reasoning,
                    )
                    if reasoning_text:
                        r_to_emit = maybe_strip_kimi_fc_substrings(
                            reasoning_text, is_kimi=is_kimi
                        )
                        if r_to_emit:
                            choice_data = ChatCompletionResponseStreamChoice(
                                index=index,
                                delta=DeltaMessage(reasoning_content=r_to_emit),
                                finish_reason=None,
                            )
                            chunk = ChatCompletionStreamResponse(
                                id=content["meta_info"]["id"],
                                created=int(time.time()),
                                choices=[choice_data],
                                model=request.model,
                            )

                            # Add usage stats if continuous_usage_stats is enabled
                            if continuous_usage_stats:
                                chunk.usage = UsageProcessor.calculate_token_usage(
                                    prompt_tokens=prompt_tokens.get(index, 0),
                                    reasoning_tokens=reasoning_tokens.get(index, 0),
                                    completion_tokens=completion_tokens.get(index, 0),
                                )

                            yield f"data: {chunk.model_dump_json()}\n\n"

                # Handle tool calls
                if (
                    request.tool_choice != "none"
                    and self.tool_call_parser
                ):
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            "%s post_reasoning_step rid=%s index=%s finish=%s "
                            "engine_delta_len=%d post_reason_delta_len=%d "
                            "post_reason_preview=%s reasoning_parser_out=%s",
                            STREAM_TOOL_DEBUG,
                            content["meta_info"].get("id"),
                            index,
                            finish_reason_type,
                            engine_text_delta_len,
                            len(delta) if delta else 0,
                            short_repr(delta),
                            short_repr(reasoning_text)
                            if reasoning_text
                            else None,
                        )
                    tool_path_chunks: List[str] = []
                    try:
                        tool_chunks_emitted = 0
                        async for chunk in iter_tool_call_stream_sse_chunks(
                            index=index,
                            delta=delta,
                            parser_dict=parser_dict,
                            content=content,
                            request=request,
                            has_tool_calls=has_tool_calls,
                            stream_tool_call_collector=stream_tool_call_collector,
                            continuous_usage_stats=continuous_usage_stats,
                            is_kimi=is_kimi,
                            tool_call_parser=self.tool_call_parser,
                        ):
                            if chunk:
                                tool_path_chunks.append(chunk)
                                tool_chunks_emitted += 1
                                yield chunk
                        if (
                            logger.isEnabledFor(logging.DEBUG)
                            and tool_chunks_emitted == 0
                            and (delta or engine_text_delta_len)
                        ):
                            logger.debug(
                                "%s tool_stream_emitted_zero_chunks rid=%s index=%s "
                                "finish=%s engine_delta_len=%d post_reason_delta_len=%d",
                                STREAM_TOOL_DEBUG,
                                content["meta_info"].get("id"),
                                index,
                                finish_reason_type,
                                engine_text_delta_len,
                                len(delta) if delta else 0,
                            )
                    except (ToolCallValidationError, ValueError):
                        # If the tool path failed before emitting anything for this
                        # delta, stream it as normal ``content`` so the client still
                        # receives text on ``length`` / ``unexpected_state``-style ends.
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug(
                                "%s tool_path_exception rid=%s index=%s "
                                "tool_path_chunks_so_far=%d delta_len=%d",
                                STREAM_TOOL_DEBUG,
                                content["meta_info"].get("id"),
                                index,
                                len(tool_path_chunks),
                                len(delta) if delta else 0,
                                exc_info=True,
                            )
                        if not tool_path_chunks:
                            to_emit = maybe_strip_kimi_fc_substrings(
                                delta, is_kimi=is_kimi
                            )
                            if to_emit:
                                yield sse_stream_plain_text_line(
                                    index=index,
                                    chatcmpl_id=content["meta_info"]["id"],
                                    model=request.model,
                                    to_emit=to_emit,
                                    choice_logprobs=choice_logprobs,
                                    continuous_usage_stats=continuous_usage_stats,
                                    usage_prompt_tokens=prompt_tokens.get(index, 0),
                                    usage_completion_tokens=completion_tokens.get(
                                        index, 0
                                    ),
                                    usage_reasoning_tokens=reasoning_tokens.get(
                                        index, 0
                                    ),
                                )
                        finish_reasons[index] = prefer_engine_finish_on_fc_leak(
                            finish_reason,
                            default_unexpected={
                                "type": "unexpected_state",
                                "matched": None,
                            },
                        )
                        has_tool_calls[index] = False
                    else:
                        # Send any remaining tool call args when generation finishes
                        if (
                            finish_reason_type is not None
                            and index in parser_dict
                        ):
                            parser = parser_dict[index]
                            try:
                                remaining_chunk = self._check_for_unstreamed_tool_args(
                                    parser,
                                    content,
                                    request,
                                    index,
                                    stream_tool_call_collector,
                                )
                            except ToolCallValidationError:
                                finish_reasons[index] = prefer_engine_finish_on_fc_leak(
                                    finish_reason,
                                    default_unexpected={
                                        "type": "unexpected_state",
                                        "matched": None,
                                    },
                                )
                                has_tool_calls[index] = False
                            else:
                                if remaining_chunk:
                                    yield remaining_chunk
                                if stream_tool_call_collector:
                                    try:
                                        stream_tool_call_collector.finalize_choice(
                                            index
                                        )
                                    except ToolCallValidationError:
                                        finish_reasons[index] = (
                                            prefer_engine_finish_on_fc_leak(
                                                finish_reason,
                                                default_unexpected={
                                                    "type": "unexpected_state",
                                                    "matched": None,
                                                },
                                            )
                                        )
                                        has_tool_calls[index] = False

                        else:
                            pass

                    # Tool path buffers JSON / markers into the parser; ``delta`` can be empty
                    # while new output tokens still produced logprobs. Emit logprob-only chunks.
                    if request.logprobs and choice_logprobs is not None:
                        choice_data = ChatCompletionResponseStreamChoice(
                            index=index,
                            delta=DeltaMessage(),
                            finish_reason=None,
                            matched_stop=None,
                            logprobs=choice_logprobs,
                        )
                        chunk = ChatCompletionStreamResponse(
                            id=content["meta_info"]["id"],
                            created=int(time.time()),
                            choices=[choice_data],
                            model=request.model,
                        )
                        if continuous_usage_stats:
                            chunk.usage = UsageProcessor.calculate_token_usage(
                                prompt_tokens=prompt_tokens.get(index, 0),
                                reasoning_tokens=reasoning_tokens.get(index, 0),
                                completion_tokens=completion_tokens.get(index, 0),
                            )
                        yield f"data: {chunk.model_dump_json()}\n\n"

                else:
                    to_emit: Optional[str] = maybe_strip_kimi_fc_substrings(
                        delta, is_kimi=is_kimi
                    )
                    # Emit when there is visible text, or when we must forward logprobs for
                    # this step (e.g. empty string delta after strip, or decode-only tokens).
                    if to_emit or (
                        request.logprobs and choice_logprobs is not None
                    ):
                        yield sse_stream_plain_text_line(
                            index=index,
                            chatcmpl_id=content["meta_info"]["id"],
                            model=request.model,
                            to_emit=to_emit,
                            choice_logprobs=choice_logprobs,
                            continuous_usage_stats=continuous_usage_stats,
                            usage_prompt_tokens=prompt_tokens.get(index, 0),
                            usage_completion_tokens=completion_tokens.get(index, 0),
                            usage_reasoning_tokens=reasoning_tokens.get(index, 0),
                        )

            # Send finish_reason chunks for each index that completed
            for idx, finish_reason_data in finish_reasons.items():
                finish_reason_type = finish_reason_data["type"]

                # Surface ``tool_calls`` when we emitted tool deltas, including when the
                # engine still reports ``length`` (avoids tool output looking "consumed"
                # by length alone).
                final_finish_reason = finish_reason_type
                if has_tool_calls.get(idx, False) and finish_reason_type in (
                    "stop",
                    "length",
                ):
                    final_finish_reason = "tool_calls"

                finish_reason_chunk = ChatCompletionStreamResponse(
                    id=content["meta_info"][
                        "id"
                    ],  # NOTE: openai uses the same chatcmpl-id for all indices
                    created=int(time.time()),
                    choices=[
                        ChatCompletionResponseStreamChoice(
                            index=idx,
                            delta=DeltaMessage(),
                            finish_reason=final_finish_reason,
                            matched_stop=(
                                finish_reason_data["matched"]
                                if "matched" in finish_reason_data
                                else None
                            ),
                        )
                    ],
                    model=request.model,
                    usage=None,
                )
                yield f"data: {finish_reason_chunk.model_dump_json()}\n\n"

            # Send hidden states if requested
            if request.return_hidden_states and hidden_states:
                for index, choice_hidden_states in hidden_states.items():
                    if choice_hidden_states:
                        last_token_hidden_states = (
                            choice_hidden_states[-1]
                            if len(choice_hidden_states) > 1
                            else []
                        )
                        hidden_states_chunk = ChatCompletionStreamResponse(
                            id=content["meta_info"]["id"],
                            created=int(time.time()),
                            choices=[
                                ChatCompletionResponseStreamChoice(
                                    index=index,
                                    delta=DeltaMessage(
                                        hidden_states=last_token_hidden_states
                                    ),
                                    finish_reason=None,  # Hidden states don't need finish_reason
                                )
                            ],
                            model=request.model,
                        )
                        yield f"data: {hidden_states_chunk.model_dump_json()}\n\n"

            if request.return_routed_experts and routed_experts:
                # Get first non-None routed_experts value
                first_routed_experts = next(
                    (v for v in routed_experts.values() if v is not None), None
                )
                if first_routed_experts is not None:
                    routed_experts_chunk = ChatCompletionStreamResponse(
                        id=content["meta_info"]["id"],
                        created=int(time.time()),
                        choices=[],  # sglext is at response level
                        model=request.model,
                        sglext=SglExt(routed_experts=first_routed_experts),
                    )
                    yield f"data: {routed_experts_chunk.model_dump_json()}\n\n"

            # Additional usage chunk
            if include_usage:
                usage = UsageProcessor.calculate_streaming_usage(
                    prompt_tokens,
                    reasoning_tokens,
                    completion_tokens,
                    cached_tokens=cached_tokens,
                    n_choices=request.n,
                    enable_cache_report=self.tokenizer_manager.server_args.enable_cache_report,
                )
                usage_chunk = ChatCompletionStreamResponse(
                    id=content["meta_info"]["id"],
                    created=int(time.time()),
                    choices=[],  # Empty choices array as per OpenAI spec
                    model=request.model,
                    usage=usage,
                )
                yield f"data: {usage_chunk.model_dump_json()}\n\n"

        except ValueError as e:
            if not stream_started:
                raise
            if isinstance(e, ToolCallValidationError):
                logger.warning(
                    "Tool call validation should be handled in the tool path: %s",
                    e,
                )
            else:
                error = self.create_streaming_error_response(str(e))
                yield f"data: {error}\n\n"

        yield "data: [DONE]\n\n"

    async def _handle_non_streaming_request(
        self,
        adapted_request: GenerateReqInput,
        request: ChatCompletionRequest,
        raw_request: Request,
    ) -> Union[ChatCompletionResponse, ErrorResponse, ORJSONResponse]:
        """Handle non-streaming chat completion request"""
        try:
            ret = await self.tokenizer_manager.generate_request(
                adapted_request, raw_request
            ).__anext__()
        except ValueError as e:
            return self.create_error_response(str(e))

        if not isinstance(ret, list):
            ret = [ret]

        response = self._build_chat_response(
            request,
            adapted_request.require_reasoning,
            ret,
            int(time.time()),
        )

        return response

    def _build_chat_response(
        self,
        request: ChatCompletionRequest,
        require_reasoning: bool,
        ret: List[Dict[str, Any]],
        created: int,
    ) -> Union[ChatCompletionResponse, ORJSONResponse]:
        """Build chat completion response from generation results"""
        choices = []

        # Build sglext at response level (from first ret_item, as these are per-request)
        first_ret = ret[0]
        routed_experts = process_routed_experts_from_ret(first_ret, request)
        cached_tokens_details = process_cached_tokens_details_from_ret(
            first_ret, request
        )
        response_sglext = None
        if routed_experts or cached_tokens_details:
            response_sglext = SglExt(
                routed_experts=routed_experts,
                cached_tokens_details=cached_tokens_details,
            )

        for idx, ret_item in enumerate(ret):
            # Process logprobs
            choice_logprobs = None
            if request.logprobs:
                choice_logprobs = self._process_response_logprobs(ret_item)

            # Handle hidden states
            hidden_states = process_hidden_states_from_ret(ret_item, request)

            is_kimi = is_kimi_k2_openai_serving(self.tool_call_parser)
            fr_raw = ret_item["meta_info"]["finish_reason"]
            finish_reason = copy.copy(fr_raw)
            text = ret_item["text"]

            # Handle reasoning content
            reasoning_text = None
            reasoning_parser = self.reasoning_parser
            if reasoning_parser and request.separate_reasoning:
                try:
                    parser = ReasoningParser(
                        model_type=reasoning_parser,
                        stream_reasoning=False,
                        force_reasoning=require_reasoning,
                        request=request,
                    )
                    reasoning_text, text = parser.parse_non_stream(text)
                except Exception as e:
                    logger.error(f"Reasoning parsing error: {e}")
                    return self.create_error_response(
                        "Failed to parse reasoning content",
                        err_type="InternalServerError",
                        status_code=500,
                    )

            # Handle tool calls
            tool_calls = None
            if (
                request.tool_choice != "none"
                and self.tool_call_parser
            ):
                history_tool_calls_cnt = self._get_history_tool_calls_cnt(request)
                # Snapshot: ``_process_tool_calls`` may raise after partial work; on failure
                # we must not discard parsed reasoning or raw model text (partial tool
                # markup/args should remain visible in ``content`` / ``reasoning_content``).
                text_before_tool_calls = text
                try:
                    tool_calls, text, finish_reason = self._process_tool_calls(
                        text_before_tool_calls,
                        request.tools or [],
                        finish_reason,
                        request.tool_choice,
                        history_tool_calls_cnt,
                    )
                except ValueError:
                    # Parse / decode failure in ``_process_tool_calls`` (not schema
                    # validation). Validation affects ``finish_reason`` only; see
                    # ``_process_tool_calls``. Restore pre-tool text for the message.
                    tool_calls = None
                    text = text_before_tool_calls
                    finish_reason = prefer_engine_finish_on_fc_leak(
                        fr_raw,
                        default_unexpected={
                            "type": "unexpected_state",
                            "matched": None,
                        },
                    )

            text, reasoning_text, tool_calls = strip_kimi_openai_choice_fields(
                is_kimi, text, reasoning_text, tool_calls
            )

            choice_data = ChatCompletionResponseChoice(
                index=idx,
                message=ChatMessage(
                    role="assistant",
                    # Do not coalesce to None on ``""``; clients need visible strings on
                    # ``length`` / ``unexpected_state`` and after partial tool parse.
                    content=text,
                    tool_calls=tool_calls,
                    reasoning_content=reasoning_text,
                ),
                logprobs=choice_logprobs,
                finish_reason=finish_reason["type"] if finish_reason else None,
                matched_stop=(
                    finish_reason["matched"]
                    if finish_reason and "matched" in finish_reason
                    else None
                ),
                hidden_states=hidden_states,
            )
            choices.append(choice_data)

        # Calculate usage
        usage = UsageProcessor.calculate_response_usage(
            ret,
            n_choices=request.n,
            enable_cache_report=self.tokenizer_manager.server_args.enable_cache_report,
        )

        return ChatCompletionResponse(
            id=ret[0]["meta_info"]["id"],
            created=created,
            model=request.model,
            choices=choices,
            usage=usage,
            metadata={"weight_version": ret[0]["meta_info"]["weight_version"]},
            sglext=response_sglext,
        )

    def _process_logprobs_tokens(
        self, logprobs: LogProbs, use_token_index: bool = False
    ) -> List[ChatCompletionTokenLogprob]:
        """Common helper to process logprobs tokens for both streaming and non-streaming

        Args:
            logprobs: LogProbs data from model
            use_token_index: True for non-streaming (use token_idx), False for streaming (use index 0)
        """
        token_logprobs = []

        for token_idx, (token, logprob) in enumerate(
            zip(logprobs.tokens, logprobs.token_logprobs)
        ):
            token_bytes = list(token.encode("utf-8"))
            top_logprobs = []
            if logprobs.top_logprobs:
                # - Non-streaming (use_token_index=True): uses token_idx for full data
                # - Streaming (use_token_index=False): uses index 0 for pre-sliced data
                top_logprobs_idx = token_idx if use_token_index else 0
                for top_token, top_logprob in logprobs.top_logprobs[
                    top_logprobs_idx
                ].items():
                    top_token_bytes = list(top_token.encode("utf-8"))
                    top_logprobs.append(
                        TopLogprob(
                            token=top_token,
                            bytes=top_token_bytes,
                            logprob=top_logprob,
                        )
                    )
            token_logprobs.append(
                ChatCompletionTokenLogprob(
                    token=token,
                    bytes=token_bytes,
                    logprob=logprob,
                    top_logprobs=top_logprobs,
                )
            )

        return token_logprobs

    def _process_response_logprobs(self, ret_item: Dict[str, Any]) -> ChoiceLogprobs:
        """Process logprobs for non-streaming response"""
        logprobs = to_openai_style_logprobs(
            output_token_logprobs=ret_item["meta_info"]["output_token_logprobs"],
            output_top_logprobs=ret_item["meta_info"].get("output_top_logprobs", None),
        )

        token_logprobs = self._process_logprobs_tokens(logprobs, use_token_index=True)
        return ChoiceLogprobs(content=token_logprobs)

    def _process_tool_call_id(
        self,
        call_item: ToolCallItem,
        history_tool_calls_cnt: int,
    ) -> str:
        """Process for generating a new and unique `tool_call_id`"""
        return format_openai_tool_call_id(
            self.tool_call_parser, call_item, history_tool_calls_cnt
        )

    def _process_tool_calls(
        self,
        text: str,
        tools: List[Any],
        finish_reason: Dict[str, Any],
        tool_choice: Optional[Union[str, ToolChoice]] = None,
        history_tool_calls_cnt: int = 0,
    ) -> ToolCallProcessingResult:
        """Process tool calls in the response.

        Schema validation never strips parsed function names or argument strings: if
        validation fails, the same parsed ``ToolCall`` objects are still returned and
        only ``finish_reason`` is adjusted (e.g. to ``unexpected_state`` for a normal
        engine ``stop``). Parse / decode errors still raise ``ValueError``.

        When the engine ends with ``length`` but we still parsed at least one tool
        call, the client-visible ``finish_reason`` is set to ``tool_calls`` (same as
        for ``stop``), so tool usage is not left under ``length`` alone.
        """
        engine_finish = copy.deepcopy(finish_reason)

        def _finish_after_validation(
            validation_ok: bool, has_parsed_tools: bool
        ) -> Dict[str, Any]:
            if not validation_ok:
                return prefer_engine_finish_on_fc_leak(
                    engine_finish,
                    default_unexpected={
                        "type": "unexpected_state",
                        "matched": None,
                    },
                )
            if (
                has_parsed_tools
                and engine_finish.get("type") in ("stop", "length")
            ):
                out = copy.deepcopy(engine_finish)
                out["type"] = "tool_calls"
                out["matched"] = None
                return out
            return copy.deepcopy(engine_finish)

        # Handle required or named tool choice
        if tool_choice == "required" or (
            isinstance(tool_choice, ToolChoice) and tool_choice.type == "function"
        ):
            try:
                tool_call_data = orjson.loads(text)
            except Exception as e:
                logger.debug(f"Rejecting generated tool call payload: {e}")
                raise ValueError(str(e)) from e

            validation_ok = True
            if self._is_generated_tool_call_validation_enabled():
                try:
                    validate_required_tool_call_payload(tool_call_data, tools)
                except ToolCallValidationError:
                    validation_ok = False

            tool_calls: List[ToolCall] = []
            for i, tool in enumerate(tool_call_data):
                arg_str = _tool_parameters_to_json_argument_str(tool.get("parameters"))
                call_info = ToolCallItem(
                    tool_index=i,  # Use the loop index as tool_index
                    name=tool["name"],
                    parameters=arg_str,
                )
                tool_id = self._process_tool_call_id(
                    call_info, history_tool_calls_cnt
                )
                tool_calls.append(
                    ToolCall(
                        id=tool_id,
                        index=i,
                        function=FunctionResponse(
                            name=tool["name"],
                            arguments=arg_str,
                        ),
                    )
                )
            return ToolCallProcessingResult(
                tool_calls,
                "",
                _finish_after_validation(validation_ok, bool(tool_calls)),
            )

        # Use parser since output is not constrained by JSON schema
        parser = FunctionCallParser(tools, self.tool_call_parser)
        if parser.has_tool_call(text):
            try:
                remaining_text, call_info_list = parser.parse_non_stream(text)
            except Exception as e:
                logger.error(f"Tool call parsing error: {e}")
                raise ValueError(str(e)) from e

            validation_ok = True
            if self._is_generated_tool_call_validation_enabled():
                try:
                    validate_parsed_tool_call_items(call_info_list, tools)
                except ToolCallValidationError:
                    validation_ok = False

            tool_calls = []
            for call_info in call_info_list:
                tool_id = self._process_tool_call_id(
                    call_info, history_tool_calls_cnt
                )
                tool_calls.append(
                    ToolCall(
                        id=tool_id,
                        index=getattr(call_info, "tool_index", None),
                        function=FunctionResponse(
                            name=call_info.name, arguments=call_info.parameters
                        ),
                    )
                )
            return ToolCallProcessingResult(
                tool_calls,
                remaining_text,
                _finish_after_validation(validation_ok, bool(tool_calls)),
            )

        return ToolCallProcessingResult(None, text, copy.deepcopy(engine_finish))

    def _process_streaming_logprobs(
        self,
        content: Dict[str, Any],
        n_prev_token: int,
        total_output_logprobs: int,
    ) -> ChoiceLogprobs:
        """Process logprobs for streaming response"""
        logprobs = to_openai_style_logprobs(
            output_token_logprobs=content["meta_info"]["output_token_logprobs"][
                n_prev_token:total_output_logprobs
            ],
            output_top_logprobs=content["meta_info"].get("output_top_logprobs", [])[
                n_prev_token:total_output_logprobs
            ],
        )

        token_logprobs = self._process_logprobs_tokens(logprobs, use_token_index=False)
        return ChoiceLogprobs(content=token_logprobs)

    def _process_reasoning_stream(
        self,
        index: int,
        delta: str,
        reasoning_parser_dict: Dict[int, ReasoningParser],
        content: Dict[str, Any],
        request: ChatCompletionRequest,
        require_reasoning: bool,
    ) -> tuple[Optional[str], str]:
        """Process reasoning content in streaming response"""
        if index not in reasoning_parser_dict:
            reasoning_parser_dict[index] = ReasoningParser(
                self.reasoning_parser,
                request.stream_reasoning,
                require_reasoning,
                request,
            )
        reasoning_parser = reasoning_parser_dict[index]
        return reasoning_parser.parse_stream_chunk(delta)

    def _inject_reasoning_chat_template_kwarg(
        self, extra_template_kwargs: Dict[str, Any], require_reasoning: bool
    ) -> None:
        """Forward serving_chat's reasoning state into chat template kwargs."""
        if self.reasoning_parser in ["deepseek-v3", "kimi_k2"]:
            extra_template_kwargs.setdefault("thinking", require_reasoning)
        elif self.reasoning_parser in ["qwen3", "glm45", "nemotron_3", "interns1", "mimo"]:
            extra_template_kwargs.setdefault("enable_thinking", require_reasoning)

    def _get_history_tool_calls_cnt(self, request: ChatCompletionRequest) -> int:
        """Counts the number of tool calls in the request's message history.

        NOTE: This method is only useful for models that include self-increasing
        history tool call idx in tool calls id, such as kimi-k2

        Args:
            request: The chat completion request object.

        Returns:
            The total number of tool calls in the history, or 0 if not applicable.
        """
        return history_tool_calls_count(request)

    def _patch_mistral_skip_special_tokens(
        self, request: ChatCompletionRequest
    ) -> None:
        """Mistral uses special tokens ([THINK]/[/THINK]) for reasoning markers,
        which get stripped when skip_special_tokens=True."""
        if (
            self.reasoning_parser in ["mistral"]
            and request.reasoning_effort is not None
            and request.reasoning_effort != "none"
        ):
            request.skip_special_tokens = False

    def _get_reasoning_from_request(self, request: ChatCompletionRequest) -> bool:
        """Judge whether the request needs reasoning for hybrid reasoning models
        NOTE: This is predefined based on model's chat template
        """
        if not self.reasoning_parser:
            return False
        if self.reasoning_parser in ["deepseek-v3"]:
            # Models that require explicit enable thinking (thinking=True)
            return (
                request.chat_template_kwargs is not None
                and request.chat_template_kwargs.get("thinking") is True
            )
        if self.reasoning_parser in ["kimi_k2"]:
            # Models that thinking by default, and can be disabled by setting thinking=False
            return (
                not request.chat_template_kwargs
                or request.chat_template_kwargs.get("thinking") is not False
            )
        if self.reasoning_parser in ["qwen3", "glm45", "nemotron_3", "interns1"]:
            # Models that thinking by default, and can be disabled by setting enable_thinking=False
            return (
                not request.chat_template_kwargs
                or request.chat_template_kwargs.get("enable_thinking") is not False
            )
        if self.reasoning_parser in ["mimo"]:
            # Models that require explicit enable thinking (enable_thinking=True)
            return (
                request.chat_template_kwargs is not None
                and request.chat_template_kwargs.get("enable_thinking") is True
            )
        if self.reasoning_parser in ["mistral"]:
            # Mistral models only reason when reasoning_effort is explicitly
            # set to a value other than None/"none" (typically "high").
            return (
                request.reasoning_effort is not None
                and request.reasoning_effort != "none"
            )
        return True  # default

    def _check_for_unstreamed_tool_args(
        self,
        parser: Union[FunctionCallParser, JsonArrayParser],
        content: Dict[str, Any],
        request: ChatCompletionRequest,
        index: int,
        stream_tool_call_collector: Optional[StreamToolCallCollector],
    ) -> Optional[str]:
        """
        Check for any remaining tool call arguments that need to be streamed
        when generation finishes. This ensures tool calls are properly completed
        even if the model generates the final arguments in the last chunk.
        """
        # Get the detector - either from FunctionCallParser or directly if json detector
        detector = parser.detector if hasattr(parser, "detector") else parser

        # Only check if we have tool calls and the detector has tracked data
        if (
            not hasattr(detector, "prev_tool_call_arr")
            or not detector.prev_tool_call_arr
        ):
            return None

        if (
            not hasattr(detector, "streamed_args_for_tool")
            or not detector.streamed_args_for_tool
        ):
            return None

        # Get the last tool call that was being processed
        tool_index = len(detector.prev_tool_call_arr) - 1
        if tool_index < 0 or tool_index >= len(detector.streamed_args_for_tool):
            return None

        # Get expected vs actual arguments
        expected_args = detector.prev_tool_call_arr[tool_index].get("arguments", {})
        expected_call = json.dumps(expected_args, ensure_ascii=False)
        actual_call = detector.streamed_args_for_tool[tool_index]

        # Check if there are remaining arguments to send
        remaining_call = (
            expected_call.replace(actual_call, "", 1)
            if actual_call in expected_call
            else ""
        )

        if remaining_call:
            emit_args = maybe_strip_remaining_tool_args(
                remaining_call, self.tool_call_parser
            )
            # Create tool call chunk with remaining arguments (emit before validation ingest
            # so a failing collector never swallows the last tool delta)
            tool_call = ToolCall(
                id=None,  # No ID for argument deltas
                index=tool_index,
                function=FunctionResponse(
                    name=None,  # No name for argument deltas
                    arguments=emit_args,
                ),
            )

            choice_data = ChatCompletionResponseStreamChoice(
                index=index,
                delta=DeltaMessage(tool_calls=[tool_call]),
                finish_reason=None,  # Don't send finish_reason with this chunk
            )

            chunk = ChatCompletionStreamResponse(
                id=content["meta_info"]["id"],
                created=int(time.time()),
                choices=[choice_data],
                model=request.model,
            )

            out = f"data: {chunk.model_dump_json()}\n\n"
            if stream_tool_call_collector:
                try:
                    stream_tool_call_collector.ingest_remaining_args(
                        choice_index=index,
                        tool_index=tool_index,
                        arguments_fragment=remaining_call,
                    )
                except ToolCallValidationError:
                    # Chunk is still valid to stream; finalize_choice will re-validate.
                    pass
            return out

        return None
