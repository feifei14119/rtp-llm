import torch
from typing import List, Dict, Any, Union
from pydantic import BaseModel

from transformers import PreTrainedTokenizerBase
from maga_transformer.utils.util import to_torch_dtype
from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters
from maga_transformer.async_decoder_engine.embedding.embedding_stream import EngineInputs, EngineOutputs

'''
用于多种多样的下游任务
'''
class CustomModule(object):
    renderer: 'CustomRenderer'
    handler: 'CustomHandler'
    def __init__(self, config: GptInitModelParameters, tokenizer: PreTrainedTokenizerBase):
        self.config_ = config
        self.tokenizer_ = tokenizer

# Class for c++
class CustomHandler(object):    
    def __init__(self, config: GptInitModelParameters):
        self.config_ = config        
    
    def tensor_info(self) -> List[str]:
        return []

    def init(self, tensor_map: Dict[str, torch.Tensor]) -> None:
        pass
    
    # 输出
    # input_ids: [token_len]
    # hidden_states: [token_len, hidden_size]
    # seq_len: [batch_size]
    # config: 根据custom model需求进行render
    
    # 输出: 
    # [batch_size], 由endpoint格式化返回
    def forward(self, input_ids: torch.Tensor, hidden_states: torch.Tensor, input_lengths: torch.Tensor, config: Dict[str, Any]) -> Union[torch.Tensor, List[Any]]:
        raise NotImplementedError
    
class CustomRenderer(object):
    def __init__(self, config: GptInitModelParameters, tokenizer: PreTrainedTokenizerBase):
        self.config_ = config
        self.tokenizer_ = tokenizer
        
    async def render_request(self, request_json: Dict[str, Any]) -> BaseModel:
        raise NotImplementedError
        
    async def create_input(self, request: BaseModel) -> EngineInputs:
        raise NotImplementedError
    
    async def render_response(self, request: BaseModel, inputs: EngineInputs, outputs: EngineOutputs) -> Dict[str, Any]:
        raise NotImplementedError
