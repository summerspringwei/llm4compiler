# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import llmserving_pb2 as llmserving__pb2


class LlmServingStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.Predict = channel.unary_unary(
                '/llmserving.LlmServing/Predict',
                request_serializer=llmserving__pb2.PredictRequest.SerializeToString,
                response_deserializer=llmserving__pb2.PredictResponse.FromString,
                )
        self.PredictStreamReply = channel.unary_stream(
                '/llmserving.LlmServing/PredictStreamReply',
                request_serializer=llmserving__pb2.PredictRequest.SerializeToString,
                response_deserializer=llmserving__pb2.PredictResponse.FromString,
                )


class LlmServingServicer(object):
    """Missing associated documentation comment in .proto file."""

    def Predict(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def PredictStreamReply(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_LlmServingServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'Predict': grpc.unary_unary_rpc_method_handler(
                    servicer.Predict,
                    request_deserializer=llmserving__pb2.PredictRequest.FromString,
                    response_serializer=llmserving__pb2.PredictResponse.SerializeToString,
            ),
            'PredictStreamReply': grpc.unary_stream_rpc_method_handler(
                    servicer.PredictStreamReply,
                    request_deserializer=llmserving__pb2.PredictRequest.FromString,
                    response_serializer=llmserving__pb2.PredictResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'llmserving.LlmServing', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class LlmServing(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def Predict(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/llmserving.LlmServing/Predict',
            llmserving__pb2.PredictRequest.SerializeToString,
            llmserving__pb2.PredictResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def PredictStreamReply(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_stream(request, target, '/llmserving.LlmServing/PredictStreamReply',
            llmserving__pb2.PredictRequest.SerializeToString,
            llmserving__pb2.PredictResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
