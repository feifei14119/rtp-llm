import os
from maga_transformer.pipeline import Pipeline
from maga_transformer.model_factory import ModelFactory
from maga_transformer.openai.openai_endpoint import OpenaiEndopoint
from maga_transformer.openai.api_datatype import ChatCompletionRequest, ChatMessage, RoleEnum
from maga_transformer.distribute.worker_info import update_master_info

import asyncio
import json
import os

os.environ['USE_NEW_DEVICE_IMPL'] = '1'
#os.environ['TEST_USING_DEVICE'] = 'CUDA'

use_rpc_model = bool(int(os.environ.get("USE_RPC_MODEL", 0)))
use_new_device = bool(int(os.environ.get("USE_NEW_DEVICE_IMPL", 0)))
using_device =os.environ.get("TEST_USING_DEVICE", 0)

print("[FEIFEI]: use_rpc_model = ", use_rpc_model)
print("[FEIFEI]: use_new_device = ", use_new_device)
print("[FEIFEI]: using_device = ", using_device)

async def main():
    update_master_info('127.0.0.1', 42345)

    #model = ModelFactory.from_huggingface("Qwen/Qwen-1_8B-Chat")
    model_config = ModelFactory.create_normal_model_config()
    model_config.use_rpc = True
    model = ModelFactory.from_huggingface(model_config.ckpt_path, model_config=model_config)
    pipeline = Pipeline(model, model.tokenizer)

    # usual request
    for res in pipeline("<|im_start|>user\nhello, what's your name<|im_end|>\n<|im_start|>assistant\n", max_new_tokens = 100):
        print(res.generate_texts)

    # openai request
    openai_endpoint = OpenaiEndopoint(model)
    messages = [
        ChatMessage(**{
            "role": RoleEnum.user,
            "content": "你是谁？",
        }),
    ]
    request = ChatCompletionRequest(messages=messages, stream=False)
    response = openai_endpoint.chat_completion(request_id=0, chat_request=request, raw_request=None)
    async for res in response:
        pass
    print((await response.gen_complete_response_once()).model_dump_json(indent=4))

    pipeline.stop()

if __name__ == '__main__':
    asyncio.run(main())
