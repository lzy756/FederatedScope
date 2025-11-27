import queue
import time
from collections import deque

from federatedscope.core.proto import gRPC_comm_manager_pb2, \
    gRPC_comm_manager_pb2_grpc


class gRPCComServeFunc(gRPC_comm_manager_pb2_grpc.gRPCComServeFuncServicer):
    def __init__(self):
        self.msg_queue = deque()

    def sendMessage(self, request, context):
        self.msg_queue.append(request)

        return gRPC_comm_manager_pb2.MessageResponse(msg='ACK')

    def receive(self, timeout=None, poll_interval=0.01):
        """
        Receive a message from the queue.

        Args:
            timeout: Maximum time to wait in seconds. None means wait forever.
            poll_interval: Time to sleep between queue checks (default 0.01s).

        Returns:
            The received message, or None if timeout is reached.
        """
        start_time = time.time()
        while len(self.msg_queue) == 0:
            if timeout is not None:
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    return None
            time.sleep(poll_interval)
        msg = self.msg_queue.popleft()
        return msg
