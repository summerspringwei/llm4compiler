import logging

import grpc
import llmserving_pb2
import llmserving_pb2_grpc

def run():
    PROMPT = '''Disassembly of section .text:\n<adpcm_coder>:\nstp x20, x19, [sp, #-0x10]!\nldrsh w8, [x3]\nldrb w9, [x3, #0x2]\ncmp w2, #0x1\nb.lt 0x40084c <adpcm_coder+0x10c>\nadrp x11, 0x400000 <write@@GLIBC_2.17+0x400000>\nadd x11, x11, #0xc9c\nldr w17, [x11, w9, uxtw #2]\nadrp x15, 0x400000 <write@@GLIBC_2.17+0x400000>\nmov w10, wzr\nmov w12, #0x1\nmov w13, #-0x8000\nmov w14, #0x7fff\nadd x15, x15, #0xe00\nmov w16, #0x58\nldrsh w18, [x0], #0x2\nasr w4, w17, #1\nmov w5, #0x2\nasr w6, w17, #2\nsubs w18, w18, w8\ncneg w7, w18, mi\ncmp w7, w17\ncsel w20, wzr, w17, lt\nsub w7, w7, w20\ncset w19, ge\ncmp w7, w17, asr #1\n<FILL_ME>\n
    '''
    with grpc.insecure_channel("localhost:50051") as channel:
        stub = llmserving_pb2_grpc.LlmServingStub(channel)
        response = stub.Predict(llmserving_pb2.PredictRequest(prompt=PROMPT))
        print("Greeter client received: " + response.prediction)

def get_stub(ip: str, port: str) -> llmserving_pb2_grpc.LlmServingStub:
    with grpc.insecure_channel(f"{ip}:{port}") as channel:
        stub = llmserving_pb2_grpc.LlmServingStub(channel)
    return stub


if __name__ == "__main__":
    logging.basicConfig()
    run()
