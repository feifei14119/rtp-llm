from typing import Optional, List, Dict, Any, Union, Callable, AsyncGenerator
import torch
import logging
from dataclasses import dataclass, field

from transformers import PreTrainedTokenizer

from maga_transformer.models.base_model import BaseTokenizer, GenerateOutput
from maga_transformer.openai.api_datatype import ChatMessage, GPTFunctionDefinition, UsageInfo, \
    ChatCompletionRequest, ChatCompletionResponseStreamChoice, DeltaMessage, FinisheReason, RoleEnum

@dataclass
class StreamResponseObject:
    choices: List[ChatCompletionResponseStreamChoice] = field(default_factory=list)
    usage: Optional[UsageInfo] = None

@dataclass
class RendererParams:
    max_seq_len: int
    eos_token_id: int
    stop_word_ids_list: List[List[int]]

@dataclass
class RenderedInputs:
    input_ids: List[int] = field(default_factory=list)
    input_images: List[str] = field(default_factory=list)

class CustomChatRenderer():
    def __init__(self,
                 tokenizer: Union[PreTrainedTokenizer, BaseTokenizer],
                 renderer_params: RendererParams,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = renderer_params.max_seq_len
        self.eos_token_id = renderer_params.eos_token_id
        self.stop_word_ids_list = renderer_params.stop_word_ids_list
        self.stop_words_list = [
            self.tokenizer.decode(stop_word_ids) for stop_word_ids in self.stop_word_ids_list
        ]
        self.extra_stop_word_ids_list: List[List[int]] = []

    def get_extra_stop_word_ids_list(self) -> List[List[int]]:
        return self.extra_stop_word_ids_list

    def render_chat(self, request: ChatCompletionRequest) -> RenderedInputs:
        raise NotImplementedError

    async def render_response_stream(
            self,
            output_generator: AsyncGenerator[GenerateOutput, None],
            request: ChatCompletionRequest,
            input_token_length: int,
    ) -> AsyncGenerator[StreamResponseObject, None]:
        index = 0
        yield StreamResponseObject(
            choices=[ChatCompletionResponseStreamChoice(
                index=index,
                delta=DeltaMessage(
                    role=RoleEnum.assistant,
                ),
            )]
        )

        output_token_length = 0
        responded_output_ids = []
        responded_string = ""
        finish_reason = None

        async for output in output_generator:
            index += 1
            output_ids = self._clean_output_ids(input_token_length, output.output_ids)
            output_token_length = len(output_ids)
            finish_reason = self._check_finish_reason(output_ids)
            output_ids = self._remove_stop_word_ids(output_ids)
            # For some tokenizers (e.g. ChatGLM), decode a single token differs from decode a list of tokens.
            decoded_prev_token = self.tokenizer.decode(responded_output_ids[-1:])
            tokens_to_decode = responded_output_ids[-1:] + output_ids[len(responded_output_ids):]
            decoded_string = self.tokenizer.decode(tokens_to_decode)
            delta_output_string = decoded_string[len(decoded_prev_token):]

            responded_output_ids = output_ids
            responded_string += delta_output_string

            if len(delta_output_string) > 0:
                responded_string += delta_output_string
                yield StreamResponseObject(
                    choices=[ChatCompletionResponseStreamChoice(
                        index=index,
                        delta=DeltaMessage(
                            content=delta_output_string,
                        ),
                    )],
                    usage=UsageInfo(
                        prompt_tokens=input_token_length,
                        total_tokens=input_token_length + output_token_length,
                        completion_tokens=output_token_length
                    )
                )

        if finish_reason == None:
            logging.debug(f"output [{responded_string}] found no stop reason! use stop as default.")
            finish_reason = FinisheReason.stop

        yield StreamResponseObject(
            choices=[ChatCompletionResponseStreamChoice(
                index=index + 1,
                delta=DeltaMessage(
                    content="",
                ),
                finish_reason=finish_reason
            )],
            usage=UsageInfo(
                prompt_tokens=input_token_length,
                total_tokens=input_token_length + output_token_length,
                completion_tokens=output_token_length
            )
        )

    def _check_finish_reason(self, token_ids: List[int]) -> Optional[FinisheReason]:
        if len(token_ids) >= self.max_seq_len:
            return FinisheReason.length
        if token_ids[-1] == self.eos_token_id:
            return FinisheReason.stop
        for stop_word_ids in self.stop_word_ids_list:
            if (len(token_ids) >= len(stop_word_ids)) and (token_ids[-len(stop_word_ids):] == stop_word_ids):
                return FinisheReason.stop
        return None

    def _remove_stop_word_ids(self, output_ids: List[int]) -> List[int]:
        for stop_word_ids in self.stop_word_ids_list:
            for i in range(1, len(stop_word_ids) + 1):
                if output_ids[-i:] == stop_word_ids[:i]:
                    output_ids = output_ids[:-i]
                    break
        return output_ids

    def _clean_output_ids(self, input_length: int, output_ids_tensor: torch.Tensor) -> list[int]:
        output_ids_tensor = output_ids_tensor.cpu().reshape([-1])
        # TODO(wangyin): This slicing shouldn't be done here.
        # model should return output length, ids should be sliced with output length.
        output_ids = output_ids_tensor[output_ids_tensor != self.eos_token_id].tolist()[input_length:]
        return output_ids

