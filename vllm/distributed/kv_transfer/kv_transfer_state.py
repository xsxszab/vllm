# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import time
from typing import TYPE_CHECKING

import torch
import zmq

from vllm.distributed.kv_transfer.kv_connector.base import KVConnectorBaseType
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1 import (
    KVConnectorBase_V1,
    KVConnectorRole,
)
from vllm.logger import init_logger
from vllm.utils.network_utils import make_zmq_socket

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig

_KV_CONNECTOR_AGENT: KVConnectorBaseType | None = None
# Worker end of the engine's early receive channel (multiprocessing executor).
_EARLY_RECV_SOCKET: zmq.Socket | None = None
# Interval between polls while a worker waits for its next step.
_EARLY_RECV_POLL_INTERVAL_S = 0.0002
# Without a step for this long the engine is idle or paused: stop polling.
_EARLY_RECV_ACTIVE_WINDOW_S = 1.0
_early_recv_last_step = 0.0
_early_recv_next_poll = 0.0

logger = init_logger(__name__)


def get_kv_transfer_group() -> KVConnectorBaseType:
    assert _KV_CONNECTOR_AGENT is not None, (
        "disaggregated KV cache transfer parallel group is not initialized"
    )
    return _KV_CONNECTOR_AGENT


def has_kv_transfer_group() -> bool:
    return _KV_CONNECTOR_AGENT is not None


def is_v1_kv_transfer_group(connector: KVConnectorBaseType | None = None) -> bool:
    """Check if the KV connector is the v1 connector.
    If the argument is None, it will check the global KV connector

    Args:
        connector: The KV connector to check. If None, it will check the
            global KV connector.

    Note:
        This function will no-longer be needed after the v1 KV connector
        becomes the default.

    """
    if connector is None:
        connector = _KV_CONNECTOR_AGENT

    if connector is None:
        return False

    return isinstance(connector, KVConnectorBase_V1)


def _sync_engine_id_across_tp(vllm_config: "VllmConfig") -> None:
    """Broadcast engine_id from TP rank 0 so all workers in a
    multi-node TP group share the same value.

    When PP is enabled, also broadcast across PP ranks so all workers in
    the same model-parallel engine share the same value.
    """
    from vllm.distributed.parallel_state import (
        get_pp_group,
        get_tp_group,
    )

    assert vllm_config.kv_transfer_config is not None
    synced_id = get_tp_group().broadcast_object(
        vllm_config.kv_transfer_config.engine_id, src=0
    )
    if vllm_config.parallel_config.pipeline_parallel_size > 1:
        synced_id = get_pp_group().broadcast_object(synced_id, src=0)
    vllm_config.kv_transfer_config.engine_id = synced_id


def ensure_kv_transfer_initialized(
    vllm_config: "VllmConfig", kv_cache_config: "KVCacheConfig"
) -> None:
    """Initialize KV cache transfer parallel group."""
    global _KV_CONNECTOR_AGENT

    if _KV_CONNECTOR_AGENT is not None:
        return

    kv_transfer_config = vllm_config.kv_transfer_config
    if kv_transfer_config is not None and kv_transfer_config.is_kv_transfer_instance:
        _sync_engine_id_across_tp(vllm_config)

        _KV_CONNECTOR_AGENT = KVConnectorFactory.create_connector(
            config=vllm_config,
            role=KVConnectorRole.WORKER,
            kv_cache_config=kv_cache_config,
        )


def init_early_recv_reporting(path: str) -> None:
    """Report KV loads that finish early to the engine over the channel at `path`."""
    global _EARLY_RECV_SOCKET, _early_recv_last_step, _early_recv_next_poll
    _EARLY_RECV_SOCKET = make_zmq_socket(zmq.Context.instance(), path, zmq.PUSH)
    _early_recv_last_step = _early_recv_next_poll = 0.0


def report_early_finished_recving(connector: KVConnectorBase_V1) -> None:
    """Send the loads `connector` finished successfully since the last report.

    Called by the model runner during each step. A no-op unless
    `init_early_recv_reporting()` was called in this process.
    """
    global _early_recv_last_step
    if _EARLY_RECV_SOCKET is None:
        return
    _early_recv_last_step = time.monotonic()
    _send_early_finished_recving(connector)


def next_early_recv_poll(connector: KVConnectorBase_V1) -> float | None:
    """`time.monotonic()` deadline of the next poll while the worker waits for
    its next step, or None when there is nothing to poll for."""
    if (
        _EARLY_RECV_SOCKET is None
        or time.monotonic() - _early_recv_last_step > _EARLY_RECV_ACTIVE_WINDOW_S
        or not connector.has_pending_recving()
    ):
        return None
    return _early_recv_next_poll


def poll_early_finished_recving(connector: KVConnectorBase_V1) -> None:
    """Poll while the worker waits for its next step; see `next_early_recv_poll()`."""
    _send_early_finished_recving(connector)


def _send_early_finished_recving(connector: KVConnectorBase_V1) -> None:
    global _early_recv_next_poll
    assert _EARLY_RECV_SOCKET is not None
    try:
        with torch.inference_mode():
            req_ids = connector.poll_finished_recving()
        if req_ids:
            _EARLY_RECV_SOCKET.send_pyobj(req_ids)
    except Exception:
        logger.exception("Early KV receive poll failed")
        # Receive state may already be lost, so no later step or fault
        # tolerance retry can complete these loads; the worker monitor stops
        # the engine.
        os._exit(1)
    finally:
        # Measured from the end of a poll, so long polls do not run back to back.
        _early_recv_next_poll = time.monotonic() + _EARLY_RECV_POLL_INTERVAL_S


def ensure_kv_transfer_shutdown() -> None:
    global _KV_CONNECTOR_AGENT, _EARLY_RECV_SOCKET
    if _KV_CONNECTOR_AGENT is not None:
        _KV_CONNECTOR_AGENT.shutdown()
        _KV_CONNECTOR_AGENT = None
    if _EARLY_RECV_SOCKET is not None:
        _EARLY_RECV_SOCKET.close(linger=0)
        _EARLY_RECV_SOCKET = None
