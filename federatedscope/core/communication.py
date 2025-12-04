import grpc
from concurrent import futures
import logging
import random
import threading
import time
import torch.distributed as dist

from collections import deque

from federatedscope.core.proto import gRPC_comm_manager_pb2, \
    gRPC_comm_manager_pb2_grpc
from federatedscope.core.gRPC_server import gRPCComServeFunc
from federatedscope.core.message import Message

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class StandaloneCommManager(object):
    """
    The communicator used for standalone mode
    """
    def __init__(self, comm_queue, monitor=None):
        self.comm_queue = comm_queue
        self.neighbors = dict()
        self.monitor = monitor  # used to track the communication related
        # metrics

    def receive(self):
        # we don't need receive() in standalone
        pass

    def add_neighbors(self, neighbor_id, address=None):
        self.neighbors[neighbor_id] = address

    def get_neighbors(self, neighbor_id=None):
        address = dict()
        if neighbor_id:
            if isinstance(neighbor_id, list):
                for each_neighbor in neighbor_id:
                    address[each_neighbor] = self.get_neighbors(each_neighbor)
                return address
            else:
                return self.neighbors[neighbor_id]
        else:
            # Get all neighbors
            return self.neighbors

    def send(self, message):
        # All the workers share one comm_queue
        self.comm_queue.append(message)


class StandaloneDDPCommManager(StandaloneCommManager):
    """
    The communicator used for standalone mode with multigpu
    """
    def __init__(self, comm_queue, monitor=None, id2comm=None):
        super().__init__(comm_queue, monitor)
        self.id2comm = id2comm
        self.device = "cuda:{}".format(dist.get_rank())

    def _send_model_para(self, model_para, dst_rank):
        for v in model_para.values():
            t = v.to(self.device)
            dist.send(tensor=t, dst=dst_rank)

    def send(self, message):
        is_model_para = message.msg_type == 'model_para'
        is_evaluate = message.msg_type == 'evaluate'
        if self.id2comm is None:
            # client to server
            if is_model_para:
                model_para = message.content[1]
                message.content = (message.content[0], {})
                self.comm_queue.append(message) if isinstance(
                    self.comm_queue, deque) else self.comm_queue.put(message)
                self._send_model_para(model_para, 0)
            else:
                self.comm_queue.append(message) if isinstance(
                    self.comm_queue, deque) else self.comm_queue.put(message)
        else:
            receiver = message.receiver
            if not isinstance(receiver, list):
                receiver = [receiver]
            if is_model_para or is_evaluate:
                model_para = message.content
                message.content = {}
            for idx, each_comm in enumerate(self.comm_queue):
                for each_receiver in receiver:
                    if each_receiver in self.neighbors and \
                            self.id2comm[each_receiver] == idx:
                        each_comm.put(message)
                        break
                if is_model_para or is_evaluate:
                    for each_receiver in receiver:
                        if each_receiver in self.neighbors and \
                                self.id2comm[each_receiver] == idx:
                            self._send_model_para(model_para, idx + 1)
                            break
        download_bytes, upload_bytes = message.count_bytes()
        self.monitor.track_upload_bytes(upload_bytes)


class gRPCCommManager(object):
    """
        The implementation of gRPCCommManager is referred to the tutorial on
        https://grpc.io/docs/languages/python/

        Enhanced with:
        - Exponential backoff retry for sending messages
        - Neighbor status tracking (online/offline)
        - Thread-safe status management
    """
    def __init__(self, host='0.0.0.0', port='50050', client_num=2, cfg=None):
        self.host = host
        self.port = port
        self.cfg = cfg
        options = [
            ("grpc.max_send_message_length", cfg.grpc_max_send_message_length),
            ("grpc.max_receive_message_length",
             cfg.grpc_max_receive_message_length),
            ("grpc.enable_http_proxy", cfg.grpc_enable_http_proxy),
        ]

        if cfg.grpc_compression.lower() == 'deflate':
            self.comp_method = grpc.Compression.Deflate
        elif cfg.grpc_compression.lower() == 'gzip':
            self.comp_method = grpc.Compression.Gzip
        else:
            self.comp_method = grpc.Compression.NoCompression

        self.server_funcs = gRPCComServeFunc()
        self.grpc_server = self.serve(max_workers=client_num,
                                      host=host,
                                      port=port,
                                      options=options)
        self.neighbors = dict()
        self.monitor = None  # used to track the communication related metrics

        # Neighbor status tracking with thread safety
        self._neighbor_status = dict()  # neighbor_id -> bool (True=online)
        self._status_lock = threading.Lock()

        # Retry configuration
        self._max_retries = getattr(cfg, 'send_max_retries', 3)
        self._retry_base_delay = getattr(cfg, 'send_retry_base_delay', 1.0)
        self._retry_max_delay = getattr(cfg, 'send_retry_max_delay', 30.0)

        # Connection lost callback (for client reconnection)
        self._connection_lost_callbacks = []

    def register_connection_lost_callback(self, callback):
        """
        Register a callback function to be called when connection is lost.
        The callback receives the neighbor_id as argument.
        """
        self._connection_lost_callbacks.append(callback)

    def _notify_connection_lost(self, neighbor_id):
        """Notify all registered callbacks that connection is lost."""
        for callback in self._connection_lost_callbacks:
            try:
                callback(neighbor_id)
            except Exception as e:
                logger.error(f'Error in connection lost callback: {e}')

    def serve(self, max_workers, host, port, options):
        """
        This function is referred to
        https://grpc.io/docs/languages/python/basics/#starting-the-server
        """
        server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=max_workers),
            compression=self.comp_method,
            options=options)
        gRPC_comm_manager_pb2_grpc.add_gRPCComServeFuncServicer_to_server(
            self.server_funcs, server)
        server.add_insecure_port("{}:{}".format(host, port))
        server.start()

        return server

    def add_neighbors(self, neighbor_id, address):
        if isinstance(address, dict):
            self.neighbors[neighbor_id] = '{}:{}'.format(
                address['host'], address['port'])
        elif isinstance(address, str):
            self.neighbors[neighbor_id] = address
        else:
            raise TypeError(f"The type of address ({type(address)}) is not "
                            "supported yet")
        # Mark new neighbor as online by default
        with self._status_lock:
            self._neighbor_status[neighbor_id] = True

    def get_neighbors(self, neighbor_id=None):
        address = dict()
        if neighbor_id:
            if isinstance(neighbor_id, list):
                for each_neighbor in neighbor_id:
                    address[each_neighbor] = self.get_neighbors(each_neighbor)
                return address
            else:
                return self.neighbors[neighbor_id]
        else:
            # Get all neighbors
            return self.neighbors

    def is_neighbor_alive(self, neighbor_id):
        """Check if a neighbor is marked as online."""
        with self._status_lock:
            return self._neighbor_status.get(neighbor_id, False)

    def mark_neighbor_offline(self, neighbor_id):
        """Mark a neighbor as offline."""
        with self._status_lock:
            if neighbor_id in self._neighbor_status:
                if self._neighbor_status[neighbor_id]:
                    logger.warning(
                        f'Neighbor #{neighbor_id} marked as OFFLINE')
                self._neighbor_status[neighbor_id] = False

    def mark_neighbor_online(self, neighbor_id):
        """Mark a neighbor as online."""
        with self._status_lock:
            if neighbor_id in self._neighbor_status:
                if not self._neighbor_status[neighbor_id]:
                    logger.info(f'Neighbor #{neighbor_id} marked as ONLINE')
                self._neighbor_status[neighbor_id] = True

    def get_alive_neighbors(self):
        """Get list of neighbor IDs that are currently online."""
        with self._status_lock:
            return [
                nid for nid, alive in self._neighbor_status.items() if alive
            ]

    def get_online_client_count(self):
        """Get the count of online neighbors."""
        with self._status_lock:
            return sum(1 for alive in self._neighbor_status.values() if alive)

    def _exponential_backoff(self, attempt):
        """Calculate delay for exponential backoff with jitter."""
        delay = min(
            self._retry_base_delay * (2**attempt) + random.uniform(0, 1),
            self._retry_max_delay)
        return delay

    def _send(self, receiver_address, message, receiver_id=None):
        """
        Send a message to a receiver with exponential backoff retry.

        Args:
            receiver_address: The address string (host:port) of the receiver
            message: The Message object to send
            receiver_id: Optional neighbor ID for status tracking
        """
        def _create_stub(receiver_address):
            """
            This part is referred to
            https://grpc.io/docs/languages/python/basics/#creating-a-stub
            """
            channel = grpc.insecure_channel(receiver_address,
                                            compression=self.comp_method,
                                            options=(('grpc.enable_http_proxy',
                                                      0), ))
            stub = gRPC_comm_manager_pb2_grpc.gRPCComServeFuncStub(channel)
            return stub, channel

        request = message.transform(to_list=True)
        last_error = None

        for attempt in range(self._max_retries):
            stub, channel = _create_stub(receiver_address)
            try:
                stub.sendMessage(request)
                channel.close()
                # Success - mark neighbor as online if we have ID
                if receiver_id is not None:
                    self.mark_neighbor_online(receiver_id)
                return True
            except grpc._channel._InactiveRpcError as error:
                last_error = error
                channel.close()
                if attempt < self._max_retries - 1:
                    delay = self._exponential_backoff(attempt)
                    logger.debug(
                        f'Send to {receiver_address} failed (attempt '
                        f'{attempt + 1}/{self._max_retries}), '
                        f'retrying in {delay:.2f}s: {error.details()}')
                    time.sleep(delay)

        # All retries failed
        logger.warning(f'Failed to send message to {receiver_address} after '
                       f'{self._max_retries} attempts: {last_error}')
        if receiver_id is not None:
            self.mark_neighbor_offline(receiver_id)
            # Notify connection lost callbacks
            self._notify_connection_lost(receiver_id)
        return False

    def send(self, message):
        """
        Send a message to receivers, skipping offline neighbors.
        """
        receiver = message.receiver
        if receiver is not None:
            if not isinstance(receiver, list):
                receiver = [receiver]
            for each_receiver in receiver:
                if each_receiver in self.neighbors:
                    # Skip offline neighbors
                    if not self.is_neighbor_alive(each_receiver):
                        logger.debug(f'Skipping send to offline neighbor '
                                     f'#{each_receiver}')
                        continue
                    receiver_address = self.neighbors[each_receiver]
                    self._send(receiver_address,
                               message,
                               receiver_id=each_receiver)
        else:
            for each_receiver in self.neighbors:
                # Skip offline neighbors
                if not self.is_neighbor_alive(each_receiver):
                    logger.debug(
                        f'Skipping send to offline neighbor #{each_receiver}')
                    continue
                receiver_address = self.neighbors[each_receiver]
                self._send(receiver_address,
                           message,
                           receiver_id=each_receiver)

    def receive(self):
        received_msg = self.server_funcs.receive()
        message = Message()
        message.parse(received_msg.msg)
        return message
