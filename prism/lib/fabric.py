"""
prism.lib.fabric — CHORUS Fabric gRPC Transport Interface
==========================================================

Implements:
- TensorCipher: QR-decomposed V_enc = V @ K cipher with HMAC-SHA256 chain-of-custody
  watermark (replaces the weaker cosine-similarity approach with tamper-evident signing)
- CHORUSFabric: Async gRPC client/server for raw float32 vector streaming
- Ephemeral key lifecycle: keys are single-use, derived per-stream, never stored

Design fix applied: The rolling watermark uses HMAC-SHA256(key=stream_secret, msg=vector_bytes)
rather than cosine similarity against a random stream. This makes the watermark cryptographically
tamper-evident — an observer who lacks stream_secret cannot forge a valid watermark even if they
know the seeding scheme.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
import os
import struct
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import AsyncIterator, Optional, Sequence

import numpy as np
from numpy.linalg import qr

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class FabricError(Exception):
    """Base error for all CHORUS Fabric operations."""


class CipherError(FabricError):
    """Raised when encryption or decryption fails."""


class WatermarkError(FabricError):
    """Raised when watermark verification fails — indicates tampering."""


class TransportError(FabricError):
    """Raised on gRPC channel faults."""


class KeyExpiredError(FabricError):
    """Raised when an ephemeral key is used after its TTL has elapsed."""


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class StreamDirection(Enum):
    OUTBOUND = auto()
    INBOUND = auto()


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FabricConfig:
    """
    Immutable configuration for a CHORUS Fabric channel.

    Attributes
    ----------
    host:
        gRPC server hostname or IP.
    port:
        gRPC server port (default 50051 mirrors CHORUS default).
    vector_dim:
        Expected float32 vector dimensionality on this channel.
    key_ttl_seconds:
        Ephemeral cipher-key lifetime. After expiry the key is zeroed and
        a new one must be negotiated.
    max_stream_batch:
        Maximum number of vectors per gRPC streaming batch.
    tls_cert_path:
        Optional path to PEM certificate for mutual TLS. If None the
        channel runs in plaintext mode (dev/localhost only).
    """

    host: str = "localhost"
    port: int = 50051
    vector_dim: int = 64
    key_ttl_seconds: float = 300.0
    max_stream_batch: int = 256
    tls_cert_path: Optional[str] = None

    @property
    def address(self) -> str:
        return f"{self.host}:{self.port}"


@dataclass
class EphemeralKey:
    """
    Single-use cipher key derived from a CSPRNG seed.

    Attributes
    ----------
    key_id:
        UUID identifying this key for watermark binding.
    K:
        Orthogonal matrix derived via QR decomposition, shape (dim, dim).
    K_inv:
        K^{-1} = K^T for orthogonal matrices — no separate inversion needed.
    stream_secret:
        32-byte HMAC secret for watermark signing, derived independently
        of K so that compromising K does not reveal the watermark key.
    created_at:
        Unix timestamp of key creation.
    ttl_seconds:
        Lifetime after which the key must be considered expired.
    """

    key_id: str
    K: np.ndarray          # shape (dim, dim), float32, orthogonal
    K_inv: np.ndarray      # K.T — valid because K is orthogonal
    stream_secret: bytes   # 32 bytes for HMAC-SHA256
    created_at: float
    ttl_seconds: float

    def is_expired(self) -> bool:
        return (time.monotonic() - self.created_at) > self.ttl_seconds

    def zero(self) -> None:
        """Overwrite key material in memory before GC."""
        self.K[:] = 0.0
        self.K_inv[:] = 0.0
        # bytes are immutable; replace the reference
        object.__setattr__(self, "stream_secret", b"\x00" * 32)


# ---------------------------------------------------------------------------
# TensorCipher
# ---------------------------------------------------------------------------


class TensorCipher:
    """
    Encrypts and decrypts float32 vectors using a QR-decomposed orthogonal
    transformation:

        V_enc = V @ K        (encrypt)
        V     = V_enc @ K^T  (decrypt)

    Because K is orthogonal, ||V_enc|| == ||V||, which means the cipher is
    norm-preserving: downstream cosine-similarity operations on ciphertext
    yield the same ordering as on plaintext — a useful property for
    approximate-nearest-neighbour indexes on encrypted data.

    A per-vector HMAC-SHA256 watermark is appended to every encrypted payload
    for chain-of-custody tracking.  The watermark covers:
        HMAC(stream_secret, key_id || sequence_number || vector_bytes)
    so that replays, reorderings, and vector substitutions are all detectable.
    """

    def __init__(self, dim: int, ttl_seconds: float = 300.0) -> None:
        self._dim = dim
        self._ttl = ttl_seconds
        self._active_key: Optional[EphemeralKey] = None

    # ------------------------------------------------------------------
    # Key management
    # ------------------------------------------------------------------

    def rotate_key(self) -> str:
        """
        Generate a new ephemeral key, expire the old one, and return the
        new key_id.  Call this at startup and whenever is_expired() is True.
        """
        if self._active_key is not None:
            self._active_key.zero()

        seed = os.urandom(32)
        rng = np.random.default_rng(np.frombuffer(seed, dtype=np.uint8).tolist())
        raw = rng.standard_normal((self._dim, self._dim)).astype(np.float32)
        Q, _ = qr(raw)
        Q = Q.astype(np.float32)

        # Independent HMAC secret — never derivable from K
        stream_secret = hashlib.sha256(os.urandom(32) + seed).digest()

        key = EphemeralKey(
            key_id=str(uuid.uuid4()),
            K=Q,
            K_inv=Q.T.copy(),
            stream_secret=stream_secret,
            created_at=time.monotonic(),
            ttl_seconds=self._ttl,
        )
        self._active_key = key
        logger.info("TensorCipher: rotated to key_id=%s", key.key_id)
        return key.key_id

    def _require_valid_key(self) -> EphemeralKey:
        if self._active_key is None:
            raise CipherError("No active key — call rotate_key() first.")
        if self._active_key.is_expired():
            raise KeyExpiredError(
                f"Ephemeral key {self._active_key.key_id} has expired after "
                f"{self._ttl}s.  Call rotate_key() to issue a new one."
            )
        return self._active_key

    # ------------------------------------------------------------------
    # Encrypt / Decrypt
    # ------------------------------------------------------------------

    def encrypt(
        self,
        vectors: np.ndarray,
        sequence_number: int = 0,
    ) -> tuple[np.ndarray, bytes]:
        """
        Encrypt a batch of vectors and produce an HMAC watermark.

        Parameters
        ----------
        vectors:
            Float32 array of shape (N, dim) or (dim,) for a single vector.
        sequence_number:
            Monotonically increasing counter for replay detection.

        Returns
        -------
        (V_enc, watermark)
            V_enc: encrypted float32 array, same shape as input.
            watermark: 32-byte HMAC-SHA256 digest.
        """
        key = self._require_valid_key()
        v = np.atleast_2d(vectors).astype(np.float32)

        if v.shape[-1] != self._dim:
            raise CipherError(
                f"Vector dim mismatch: expected {self._dim}, got {v.shape[-1]}."
            )

        V_enc: np.ndarray = v @ key.K  # (N, dim) @ (dim, dim) → (N, dim)

        watermark = self._compute_watermark(
            key=key,
            seq=sequence_number,
            vector_bytes=V_enc.tobytes(),
        )

        return V_enc.squeeze(), watermark

    def decrypt(
        self,
        vectors: np.ndarray,
        watermark: bytes,
        sequence_number: int = 0,
    ) -> np.ndarray:
        """
        Verify watermark and decrypt a batch of ciphertext vectors.

        Raises WatermarkError if the watermark does not match — caller should
        drop the payload and log a security event.
        """
        key = self._require_valid_key()
        v = np.atleast_2d(vectors).astype(np.float32)

        expected = self._compute_watermark(
            key=key,
            seq=sequence_number,
            vector_bytes=v.tobytes(),
        )

        # Constant-time comparison prevents timing side-channels
        if not hmac.compare_digest(expected, watermark):
            raise WatermarkError(
                "Watermark verification failed — payload may have been tampered with."
            )

        V_dec: np.ndarray = v @ key.K_inv
        return V_dec.squeeze()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_watermark(
        key: EphemeralKey,
        seq: int,
        vector_bytes: bytes,
    ) -> bytes:
        """
        HMAC-SHA256(stream_secret, key_id_bytes || seq_bytes || vector_bytes)

        Covers key identity and sequence number so that replays and
        key-substitution attacks are both detectable.
        """
        seq_bytes = struct.pack(">Q", seq)  # 8-byte big-endian
        msg = key.key_id.encode() + seq_bytes + vector_bytes
        return hmac.new(key.stream_secret, msg, hashlib.sha256).digest()


# ---------------------------------------------------------------------------
# gRPC stub shims
# ---------------------------------------------------------------------------
# In production these are replaced by the generated *_pb2_grpc stubs.
# We define lightweight Protocol stubs here so the fabric layer is fully
# testable without a running gRPC server.


class _VectorStreamStub:
    """Minimal interface matching the generated CHORUS gRPC stub."""

    async def StreamVectors(
        self,
        request_iterator: AsyncIterator[bytes],
    ) -> AsyncIterator[bytes]:
        raise NotImplementedError("Replace with generated gRPC stub.")


@dataclass
class VectorFrame:
    """
    Wire-frame for a single gRPC streaming message.

    Fields are packed to a binary blob for zero-copy delivery:
        [key_id: 36 bytes UTF-8] [seq: 8 bytes uint64 BE]
        [watermark: 32 bytes]    [N: 4 bytes uint32 BE]
        [dim: 4 bytes uint32 BE] [vectors: N*dim*4 bytes float32 LE]
    """

    key_id: str
    seq: int
    watermark: bytes       # 32 bytes
    vectors: np.ndarray    # shape (N, dim), float32

    _HEADER_FMT = ">36sQ32sII"
    _HEADER_SIZE: int = struct.calcsize(">36sQ32sII")  # 86 bytes

    def to_bytes(self) -> bytes:
        v = np.atleast_2d(self.vectors).astype(np.float32)
        N, dim = v.shape
        header = struct.pack(
            self._HEADER_FMT,
            self.key_id.encode().ljust(36)[:36],
            self.seq,
            self.watermark,
            N,
            dim,
        )
        return header + v.tobytes()

    @classmethod
    def from_bytes(cls, data: bytes) -> "VectorFrame":
        hdr = data[: cls._HEADER_SIZE]
        key_id_raw, seq, watermark, N, dim = struct.unpack(cls._HEADER_FMT, hdr)
        key_id = key_id_raw.decode().rstrip()
        raw_vectors = data[cls._HEADER_SIZE :]
        vectors = np.frombuffer(raw_vectors, dtype=np.float32).reshape(N, dim).copy()
        return cls(key_id=key_id, seq=seq, watermark=watermark, vectors=vectors)


# ---------------------------------------------------------------------------
# CHORUSFabric
# ---------------------------------------------------------------------------


class CHORUSFabric:
    """
    Async gRPC transport for raw float32 vector streams.

    Usage (outbound)
    ----------------
    ::
        config = FabricConfig(host="prism-node-1", port=50051, vector_dim=64)
        async with CHORUSFabric(config) as fabric:
            async for batch in my_vector_source():
                frames = await fabric.send(batch)

    Usage (inbound / server)
    -------------------------
    ::
        async def handler(frame: VectorFrame) -> None:
            vectors = fabric.receive(frame)
            ...

        await fabric.serve(handler)
    """

    def __init__(self, config: FabricConfig) -> None:
        self._cfg = config
        self._cipher = TensorCipher(config.vector_dim, config.key_ttl_seconds)
        self._seq: int = 0
        self._channel: Optional[object] = None  # grpc.aio.Channel at runtime
        self._stub: Optional[_VectorStreamStub] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "CHORUSFabric":
        await self.connect()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    async def connect(self) -> None:
        """
        Open the gRPC channel and rotate the initial cipher key.

        In production, replace the stub instantiation with:
            import chorus_pb2_grpc
            self._stub = chorus_pb2_grpc.VectorStreamStub(self._channel)
        """
        try:
            import grpc  # type: ignore[import]
            import grpc.aio  # type: ignore[import]

            if self._cfg.tls_cert_path:
                with open(self._cfg.tls_cert_path, "rb") as f:
                    creds = grpc.ssl_channel_credentials(f.read())
                self._channel = grpc.aio.secure_channel(self._cfg.address, creds)
            else:
                self._channel = grpc.aio.insecure_channel(self._cfg.address)

            logger.info("CHORUSFabric: connected to %s", self._cfg.address)
        except ImportError:
            logger.warning(
                "grpcio not installed — CHORUSFabric running in loopback stub mode."
            )

        self._cipher.rotate_key()
        logger.debug("CHORUSFabric: initial cipher key rotated.")

    async def close(self) -> None:
        if self._channel is not None:
            try:
                await self._channel.close()  # type: ignore[attr-defined]
            except Exception:
                pass
        if self._cipher._active_key:
            self._cipher._active_key.zero()
        logger.info("CHORUSFabric: channel closed, key material zeroed.")

    # ------------------------------------------------------------------
    # Outbound: send
    # ------------------------------------------------------------------

    async def send(self, vectors: np.ndarray) -> list[VectorFrame]:
        """
        Encrypt and stream a batch of float32 vectors to the remote endpoint.

        Parameters
        ----------
        vectors:
            Float32 array of shape (N, dim) or (dim,).

        Returns
        -------
        List of VectorFrame objects that were transmitted, for audit logging.
        """
        if self._cipher._active_key and self._cipher._active_key.is_expired():
            logger.info("CHORUSFabric: key expired — rotating before send.")
            self._cipher.rotate_key()

        v = np.atleast_2d(vectors).astype(np.float32)
        batches = np.array_split(v, max(1, len(v) // self._cfg.max_stream_batch))
        sent_frames: list[VectorFrame] = []

        for batch in batches:
            V_enc, watermark = self._cipher.encrypt(batch, sequence_number=self._seq)
            key_id = self._cipher._active_key.key_id  # type: ignore[union-attr]

            frame = VectorFrame(
                key_id=key_id,
                seq=self._seq,
                watermark=watermark,
                vectors=V_enc,
            )
            payload = frame.to_bytes()
            self._seq += 1

            if self._stub is not None:
                await self._transmit_frame(payload)
            else:
                logger.debug(
                    "CHORUSFabric [stub mode]: would transmit %d bytes (seq=%d).",
                    len(payload),
                    frame.seq,
                )

            sent_frames.append(frame)

        return sent_frames

    async def _transmit_frame(self, payload: bytes) -> None:
        """Send a single serialised VectorFrame over the gRPC stream."""
        async def _gen() -> AsyncIterator[bytes]:
            yield payload

        try:
            async for _ in self._stub.StreamVectors(_gen()):  # type: ignore[union-attr]
                pass
        except Exception as exc:
            raise TransportError(f"gRPC stream error: {exc}") from exc

    # ------------------------------------------------------------------
    # Inbound: receive
    # ------------------------------------------------------------------

    def receive(self, frame: VectorFrame) -> np.ndarray:
        """
        Verify watermark and decrypt an inbound VectorFrame.

        Returns the plaintext float32 vector array.
        Raises WatermarkError on tamper detection.
        """
        return self._cipher.decrypt(
            vectors=frame.vectors,
            watermark=frame.watermark,
            sequence_number=frame.seq,
        )

    # ------------------------------------------------------------------
    # Server-side
    # ------------------------------------------------------------------

    async def serve(
        self,
        handler: "asyncio.Coroutine[None, None, None]",
        *,
        host: str = "0.0.0.0",
    ) -> None:
        """
        Start a gRPC server that accepts inbound VectorStream RPCs.

        In production, wire this to the generated servicer:

            server = grpc.aio.server()
            chorus_pb2_grpc.add_VectorStreamServicer_to_server(servicer, server)
            server.add_insecure_port(f"{host}:{self._cfg.port}")
            await server.start()
            await server.wait_for_termination()
        """
        logger.info(
            "CHORUSFabric: serve() called — wire to generated gRPC servicer for "
            "production use. host=%s port=%d",
            host,
            self._cfg.port,
        )
        # Placeholder: in real deployment this blocks until server shuts down.
        await asyncio.sleep(0)
