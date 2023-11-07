"""The Python implementation of the GRPC LlmServing.Predict server."""

from concurrent import futures
import logging
import grpc

from transformers import (
    CodeLlamaTokenizer,
    LlamaForCausalLM
)

import llmserving_pb2, llmserving_pb2_grpc
from run_sft_predict import get_sft_tokenizer_and_model, run_model_inference
from run_demo_codellama import get_codellama_tokenizer_and_model

from custom_logging import get_custom_logger
logger = get_custom_logger()

class LlamaServer(llmserving_pb2_grpc.LlmServingServicer):
    def __init__(self, tokenizer: CodeLlamaTokenizer, model: LlamaForCausalLM):
        self.tokenizer = tokenizer
        self.model = model
        self.count = 0
    

    def Predict(self, request, context):
        output = run_model_inference(self.tokenizer, self.model, request.prompt)
        logger.info(f"recieved request from clinet, total count {self.count}")
        self.count += 1
        
        return llmserving_pb2.PredictResponse(prediction=output)


def serve(tokenizer, model):
    """Given the tokenizer and model, run a LLM service
    """
    port = "50051"
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    llmserving_pb2_grpc.add_LlmServingServicer_to_server(LlamaServer(tokenizer, model), server)
    server.add_insecure_port("[::]:" + port)
    server.start()
    print("Server started, listening on " + port)
    server.wait_for_termination()


def serve_sft():
    """Serve the SFT model
    """
    tokenizer, model = get_sft_tokenizer_and_model()
    serve(tokenizer, model)


def serve_codellama():
    """Serve the CodeLlama model
    """
    pretrain_model = "/home2/xiachunwei/Dataset/CodeLlama-7b-Instruct-hf"
    tokenizer, model = get_codellama_tokenizer_and_model(pretrain_model)
    serve(tokenizer, model)


if __name__ == "__main__":
    logging.basicConfig()
    serve_codellama()
