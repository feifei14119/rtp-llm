import logging.config
from typing import Any, Dict

from fastapi import FastAPI
from fastapi import Request as RawRequest
from maga_transformer.server.inference_server import InferenceServer
from maga_transformer.server.misc import check_is_master


def register_embedding_api(app: FastAPI, inference_server: InferenceServer):
    # 通过路径区别请求的方式，为后续可能存在的多种task类型做后向兼容
    @app.post("/v1/embeddings")    
    @check_is_master()
    async def embedding(request: Dict[str, Any], raw_request: RawRequest):
        return await inference_server.embedding(request, raw_request)
    
    @app.post("/v1/embeddings/similarity")
    @check_is_master()
    async def similarity(request: Dict[str, Any], raw_request: RawRequest):
        return await inference_server.embedding(request, raw_request)

    @app.post("/v1/classifier")
    @check_is_master()
    async def classifier(request: Dict[str, Any], raw_request: RawRequest):
        return await inference_server.embedding(request, raw_request)
    
    @app.post("/v1/reranker")
    @check_is_master()
    async def reranker(request: Dict[str, Any], raw_request: RawRequest):
        return await inference_server.embedding(request, raw_request)