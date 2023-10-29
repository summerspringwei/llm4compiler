"""The Python implementation of the GRPC LlmServing.Predict server."""

from concurrent import futures
import logging
import grpc

from transformers import (
    CodeLlamaTokenizer,
    LlamaForCausalLM
)

import llmserving_pb2, llmserving_pb2_grpc
from run_sft_predict import get_tokenizer_and_model, run_model_inference
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


def serve():
    tokenizer, model = get_tokenizer_and_model()
    port = "50051"
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    llmserving_pb2_grpc.add_LlmServingServicer_to_server(LlamaServer(tokenizer, model), server)
    server.add_insecure_port("[::]:" + port)
    server.start()
    print("Server started, listening on " + port)
    server.wait_for_termination()


if __name__ == "__main__":
    logging.basicConfig()
    serve()