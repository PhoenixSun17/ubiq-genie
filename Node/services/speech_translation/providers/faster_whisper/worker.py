"""Local, explicitly scheduled ASR worker for Ubiq translation.

stdin/stdout are bounded UTF-8 JSON lines. Audio messages contain base64 PCM16LE;
completed utterances stay in bounded memory until Node sends ``transcribe`` or
``discard``. The input/VAD thread never performs Whisper inference. One separate
thread serves explicit ASR requests while input continues to be consumed.

Only standard-library modules are imported at module scope so protocol and
segmentation tests can supply fake models without importing GPU dependencies.
"""

from __future__ import annotations

import argparse
from array import array
import base64
import binascii
from collections import deque
from contextlib import redirect_stdout
from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import queue
import sys
import threading
import time
from typing import Any, BinaryIO, Callable, Iterable, TextIO
import uuid


SAMPLE_RATE = 16000
VAD_FRAME_SAMPLES = 512  # Silero v6: 32 ms at 16 kHz.
MAX_LINE_BYTES = 256 * 1024
MAX_AUDIO_MS = 250
MAX_ID_LENGTH = 128
MAX_TRANSCRIPT_CHARS = 8192
_DLL_HANDLES: list[Any] = []


def bounded_integer(value: Any, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or not 1 <= len(value) <= MAX_ID_LENGTH:
        raise ValueError(f"{name} must be a nonempty string of at most {MAX_ID_LENGTH} characters")
    if any(ord(character) < 32 for character in value):
        raise ValueError(f"{name} must not contain control characters")
    return value


@dataclass(frozen=True)
class VadConfig:
    silence_ms: int = 700
    pre_roll_ms: int = 300
    min_speech_ms: int = 300
    max_speech_ms: int = 15000
    threshold: float = 0.5

    @classmethod
    def from_dict(cls, value: Any) -> VadConfig:
        if not isinstance(value, dict):
            raise ValueError("vad must be an object")
        silence = bounded_integer(value.get("silenceMs", 700), "vad.silenceMs", 32, 5000)
        pre_roll = bounded_integer(value.get("preRollMs", 300), "vad.preRollMs", 0, 2000)
        minimum = bounded_integer(value.get("minSpeechMs", 300), "vad.minSpeechMs", 32, 5000)
        maximum = bounded_integer(value.get("maxSpeechMs", 15000), "vad.maxSpeechMs", 1000, 15000)
        threshold = value.get("threshold", 0.5)
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise ValueError("vad.threshold must be a number")
        if not math.isfinite(threshold) or not 0 < threshold < 1:
            raise ValueError("vad.threshold must be between zero and one")
        if pre_roll + minimum >= maximum or silence >= maximum:
            raise ValueError("vad timings must leave space for speech within maxSpeechMs")
        return cls(silence, pre_roll, minimum, maximum, float(threshold))


@dataclass(frozen=True)
class WorkerConfig:
    model_path: str
    device: str = "cuda"
    compute_type: str = "float16"
    vad: VadConfig = field(default_factory=VadConfig)
    max_pending_utterances: int = 9
    max_peers: int = 32

    @classmethod
    def from_dict(cls, value: Any) -> WorkerConfig:
        if not isinstance(value, dict):
            raise ValueError("config must be an object")
        model_path = value.get("modelPath")
        if not isinstance(model_path, str) or not model_path.strip():
            raise ValueError("modelPath must point to the downloaded local model directory")
        device = value.get("device", "cuda")
        compute_type = value.get("computeType", "float16")
        if device != "cuda" or compute_type != "float16":
            raise ValueError("this worker requires device=cuda and computeType=float16")
        return cls(
            model_path=model_path,
            device=device,
            compute_type=compute_type,
            vad=VadConfig.from_dict(value.get("vad", {})),
            max_pending_utterances=bounded_integer(
                value.get("maxPendingUtterances", 9), "maxPendingUtterances", 1, 32
            ),
            max_peers=bounded_integer(value.get("maxPeers", 32), "maxPeers", 1, 32),
        )


class JsonEmitter:
    """Serialize protocol records atomically across the reader and ASR threads."""

    def __init__(self, stream: TextIO):
        self.stream = stream
        self.lock = threading.Lock()

    def __call__(self, message: dict[str, Any]) -> None:
        payload = json.dumps(message, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        with self.lock:
            self.stream.write(payload + "\n")
            self.stream.flush()


class UtteranceSegmenter:
    """Bounded per-peer audio segmentation, independently testable without models.

    Silence timing is quantized to Silero's 32 ms frames. maxSpeechMs bounds the
    complete retained waveform, including preroll and trailing silence. Forced
    cuts do not duplicate samples or lose the remainder of the current frame.
    """

    def __init__(self, config: VadConfig, on_utterance: Callable[[array], None]):
        self.config = config
        self.on_utterance = on_utterance
        self.max_samples = config.max_speech_ms * SAMPLE_RATE // 1000
        self.min_samples = config.min_speech_ms * SAMPLE_RATE // 1000
        self.silence_limit = config.silence_ms * SAMPLE_RATE // 1000
        self.pre_roll = deque(maxlen=config.pre_roll_ms * SAMPLE_RATE // 1000)
        self.active: array | None = None
        self.speech_samples = 0
        self.silent_samples = 0

    def feed(self, samples: Iterable[float], probability: float) -> None:
        frame = array("f", samples)
        if not 0 < len(frame) <= VAD_FRAME_SAMPLES:
            raise ValueError("VAD frame must contain 1 to 512 samples")
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("VAD returned an invalid probability")
        cutoff = self.config.threshold if self.active is None else max(0.01, self.config.threshold - 0.15)
        is_speech = probability >= cutoff
        if self.active is None and not is_speech:
            self.pre_roll.extend(frame)
            return

        offset = 0
        while offset < len(frame):
            if self.active is None:
                if not is_speech:
                    self.pre_roll.extend(frame[offset:])
                    break
                self.active = array("f", self.pre_roll)
                self.pre_roll.clear()
                self.speech_samples = 0
                self.silent_samples = 0

            take = min(len(frame) - offset, self.max_samples - len(self.active))
            self.active.extend(frame[offset : offset + take])
            offset += take
            if is_speech:
                self.speech_samples += take
                self.silent_samples = 0
            else:
                self.silent_samples += take

            if len(self.active) >= self.max_samples:
                self._finish(keep_silence_preroll=False)
            elif self.silent_samples >= self.silence_limit:
                self._finish(keep_silence_preroll=True)

    def _finish(self, keep_silence_preroll: bool) -> None:
        completed = self.active
        enough_speech = self.speech_samples >= self.min_samples
        self.active = None
        self.speech_samples = 0
        self.silent_samples = 0
        self.pre_roll.clear()
        if completed is None:
            return
        if keep_silence_preroll and self.pre_roll.maxlen:
            self.pre_roll.extend(completed[-self.pre_roll.maxlen :])
        if enough_speech:
            self.on_utterance(completed)


class PeerAudioStream:
    """A stateful resampler, frame accumulator, VAD state, and segmenter per peer."""

    def __init__(
        self,
        sample_rate: int,
        channels: int,
        resampler: Callable[[bytes], Iterable[float]],
        vad: Callable[[array], float],
        segmenter: UtteranceSegmenter,
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.resampler = resampler
        self.vad = vad
        self.segmenter = segmenter
        self.frame_buffer = array("f")

    def feed(self, pcm: bytes) -> None:
        self.frame_buffer.extend(self.resampler(pcm))
        offset = 0
        while len(self.frame_buffer) - offset >= VAD_FRAME_SAMPLES:
            frame = self.frame_buffer[offset : offset + VAD_FRAME_SAMPLES]
            self.segmenter.feed(frame, float(self.vad(frame)))
            offset += VAD_FRAME_SAMPLES
        if offset:
            self.frame_buffer = self.frame_buffer[offset:]


@dataclass
class PendingUtterance:
    request_id: str
    peer_uuid: str
    samples: array
    ended_at: int
    cancelled: bool = False


class TranslationWorker:
    def __init__(
        self,
        config: WorkerConfig,
        emit: Callable[[dict[str, Any]], None],
        peer_factory: Callable[[int, int, Callable[[array], None]], PeerAudioStream],
        transcriber: Callable[[array], dict[str, Any]],
        clock_ms: Callable[[], int] | None = None,
    ):
        self.config = config
        self.emit = emit
        self.peer_factory = peer_factory
        self.transcriber = transcriber
        self.clock_ms = clock_ms or (lambda: time.time_ns() // 1_000_000)
        self.peers: dict[str, PeerAudioStream] = {}
        self.pending: dict[str, PendingUtterance] = {}
        self.active: PendingUtterance | None = None
        self.lock = threading.Lock()
        self.jobs: queue.Queue[PendingUtterance | None] = queue.Queue(maxsize=1)
        self.stopping = threading.Event()
        self.thread = threading.Thread(target=self._asr_loop, name="local-asr", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def error(self, message: str, **context: Any) -> None:
        self.emit({"type": "error", **context, "message": message[:2048]})

    def handle(self, message: Any) -> bool:
        """Handle one command; return False only for an explicit shutdown."""
        context: dict[str, Any] = {}
        try:
            if not isinstance(message, dict):
                raise ValueError("command must be an object")
            for key in ("peerUuid", "requestId"):
                if key in message:
                    context[key] = identifier(message[key], key)
            command = message.get("type")
            if command == "audio":
                self._audio(message)
            elif command == "transcribe":
                self._transcribe(identifier(message.get("requestId"), "requestId"))
            elif command == "discard":
                self._discard(identifier(message.get("requestId"), "requestId"))
            elif command == "reset_peer":
                # Backpressure recovery: completed and active ASR jobs survive.
                self.peers.pop(identifier(message.get("peerUuid"), "peerUuid"), None)
            elif command == "remove_peer":
                self._remove_peer(identifier(message.get("peerUuid"), "peerUuid"))
            elif command == "shutdown":
                self.close()
                return False
            else:
                raise ValueError("unknown command type")
        except (ValueError, TypeError, RuntimeError, binascii.Error) as exc:
            self.error(str(exc), **context)
        return True

    def _audio(self, message: dict[str, Any]) -> None:
        peer_uuid = identifier(message.get("peerUuid"), "peerUuid")
        sample_rate = bounded_integer(message.get("sampleRate"), "sampleRate", 8000, 96000)
        channels = bounded_integer(message.get("channels"), "channels", 1, 2)
        encoded = message.get("samples")
        max_bytes = sample_rate * channels * 2 * MAX_AUDIO_MS // 1000
        if not isinstance(encoded, str) or not encoded or len(encoded) > ((max_bytes + 2) // 3) * 4:
            raise ValueError(f"samples must contain at most {MAX_AUDIO_MS} ms of base64 PCM16LE")
        pcm = base64.b64decode(encoded, validate=True)
        if not pcm or len(pcm) > max_bytes or len(pcm) % (2 * channels):
            raise ValueError("samples must contain complete PCM16LE channel frames within the audio limit")
        peer = self.peers.get(peer_uuid)
        if peer is not None and (peer.sample_rate != sample_rate or peer.channels != channels):
            # Track renegotiation drops only unfinished audio and recurrent VAD state.
            # Completed and active utterances already own their 16 kHz waveform.
            self.peers.pop(peer_uuid)
            peer = None
        if peer is None:
            if len(self.peers) >= self.config.max_peers:
                raise ValueError("maximum active peer count reached")
            peer = self.peer_factory(
                sample_rate,
                channels,
                lambda samples: self._retain_utterance(peer_uuid, samples),
            )
            self.peers[peer_uuid] = peer
        try:
            peer.feed(pcm)
        except Exception as exc:
            # Never continue with partially advanced resampler or recurrent state.
            self.peers.pop(peer_uuid, None)
            raise RuntimeError(f"audio processing failed; live peer state reset: {exc}") from exc

    def _retain_utterance(self, peer_uuid: str, samples: array) -> None:
        request_id = str(uuid.uuid4())
        completed = PendingUtterance(request_id, peer_uuid, samples, self.clock_ms())
        with self.lock:
            count = len(self.pending) + int(self.active is not None)
            accepted = not self.stopping.is_set() and count < self.config.max_pending_utterances
            if accepted:
                self.pending[request_id] = completed
        if not accepted:
            self.error(
                "pending utterance limit reached; new utterance dropped",
                requestId=request_id,
                peerUuid=peer_uuid,
            )
            return
        self.emit({
            "type": "utterance",
            "requestId": request_id,
            "peerUuid": peer_uuid,
            "durationMs": round(len(samples) * 1000 / SAMPLE_RATE, 3),
            "endedAt": completed.ended_at,
        })

    def _transcribe(self, request_id: str) -> None:
        with self.lock:
            if self.stopping.is_set():
                raise ValueError("worker is shutting down")
            if self.active is not None:
                raise ValueError("an ASR request is already active; wait for its terminal response")
            utterance = self.pending.pop(request_id, None)
            if utterance is None:
                raise ValueError("utterance does not exist or was discarded")
            self.active = utterance
            self.jobs.put_nowait(utterance)

    def _discard(self, request_id: str) -> None:
        with self.lock:
            self.pending.pop(request_id, None)
            if self.active is not None and self.active.request_id == request_id:
                # Native inference cannot be interrupted safely. Retain the busy
                # guard until it returns, then emit a terminal cancellation.
                self.active.cancelled = True

    def _remove_peer(self, peer_uuid: str) -> None:
        self.peers.pop(peer_uuid, None)
        with self.lock:
            removed = [job for job in self.pending.values() if job.peer_uuid == peer_uuid]
            for job in removed:
                self.pending.pop(job.request_id, None)
            if self.active is not None and self.active.peer_uuid == peer_uuid:
                self.active.cancelled = True
        for job in removed:
            self.error("peer removed; utterance discarded", requestId=job.request_id, peerUuid=peer_uuid)

    def _asr_loop(self) -> None:
        while not self.stopping.is_set():
            try:
                job = self.jobs.get(timeout=0.25)
            except queue.Empty:
                continue
            if job is None:
                return
            result: dict[str, Any] | None = None
            failure: str | None = None
            try:
                if not job.cancelled:
                    result = self.transcriber(job.samples)
            except Exception as exc:
                failure = f"ASR failed: {exc}"
            # The loop's local job survives while waiting for its next command;
            # release its waveform now instead of retaining an extra utterance.
            job.samples = array("f")
            # Linearize peer removal with the terminal record. A new Node
            # request may arrive immediately; its reader waits for this lock
            # until the current terminal record has been fully emitted.
            with self.lock:
                cancelled = job.cancelled
                self.active = None
                if self.stopping.is_set():
                    return
                if cancelled:
                    self.error("ASR request canceled", requestId=job.request_id, peerUuid=job.peer_uuid)
                elif failure is not None:
                    self.error(failure, requestId=job.request_id, peerUuid=job.peer_uuid)
                elif result is not None:
                    self.emit({
                        "type": "transcript",
                        "requestId": job.request_id,
                        "peerUuid": job.peer_uuid,
                        **result,
                    })
                else:
                    self.error("ASR returned no result", requestId=job.request_id, peerUuid=job.peer_uuid)

    def close(self) -> None:
        self.stopping.set()
        self.peers.clear()
        with self.lock:
            self.pending.clear()
            if self.active is not None:
                self.active.cancelled = True
        try:
            self.jobs.put_nowait(None)
        except queue.Full:
            pass
        if self.thread.is_alive() and self.thread is not threading.current_thread():
            self.thread.join(timeout=1.0)


def load_runtime(config: WorkerConfig) -> tuple[Callable[..., PeerAudioStream], Callable[[array], dict[str, Any]]]:
    """Load one GPU Whisper model and one CPU Silero ONNX session, offline."""
    if os.name == "nt":
        for directory in os.environ.get("TRANSLATION_CUDA_DLL_DIRS", "").split(os.pathsep):
            if not directory:
                continue
            candidate = Path(directory)
            if not candidate.is_absolute() or not candidate.is_dir():
                raise ValueError("TRANSLATION_CUDA_DLL_DIRS must contain existing absolute directories")
            # Keep handles alive for delayed CUDA/cuDNN loads during inference.
            _DLL_HANDLES.append(os.add_dll_directory(str(candidate)))
    # These imports intentionally occur only after configuration validation.
    import ctranslate2
    import numpy as np
    import onnxruntime as ort
    import soxr
    from faster_whisper import WhisperModel
    from faster_whisper.utils import get_assets_path

    model_path = Path(config.model_path).expanduser().resolve(strict=True)
    required_files = ("model.bin", "config.json", "tokenizer.json", "preprocessor_config.json")
    missing = [name for name in required_files if not (model_path / name).is_file()]
    if not model_path.is_dir() or missing:
        raise ValueError("local Faster-Whisper model is incomplete; missing: " + ", ".join(missing))
    if ctranslate2.get_cuda_device_count() < 1:
        raise RuntimeError("CUDA device unavailable; this app does not silently fall back to CPU ASR")
    if "float16" not in ctranslate2.get_supported_compute_types("cuda"):
        raise RuntimeError("CUDA float16 inference is unavailable in this CTranslate2 installation")

    options = ort.SessionOptions()
    options.inter_op_num_threads = 1
    options.intra_op_num_threads = 1
    options.enable_cpu_mem_arena = False
    vad_path = Path(get_assets_path()) / "silero_vad_v6.onnx"
    if not vad_path.is_file():
        raise RuntimeError("Faster-Whisper's bundled silero_vad_v6.onnx is missing; reinstall the pinned requirements")
    session = ort.InferenceSession(str(vad_path), sess_options=options, providers=["CPUExecutionProvider"])
    if {entry.name for entry in session.get_inputs()} != {"input", "h", "c"}:
        raise RuntimeError("unexpected Silero ONNX interface; use the pinned Faster-Whisper version")
    print(f"Loading local ASR model on {config.device} ({config.compute_type}): {model_path}", file=sys.stderr)
    model = WhisperModel(
        str(model_path),
        device=config.device,
        compute_type=config.compute_type,
        local_files_only=True,
        num_workers=1,
    )

    def peer_factory(sample_rate: int, channels: int, on_utterance: Callable[[array], None]) -> PeerAudioStream:
        stream = None if sample_rate == SAMPLE_RATE else soxr.ResampleStream(
            sample_rate, SAMPLE_RATE, 1, dtype="float32", quality="HQ"
        )
        h = np.zeros((1, 1, 128), dtype=np.float32)
        c = np.zeros((1, 1, 128), dtype=np.float32)
        context = np.zeros((1, 64), dtype=np.float32)

        def resample(pcm: bytes) -> Iterable[float]:
            mono = np.frombuffer(pcm, dtype="<i2").astype(np.float32).reshape(-1, channels)
            mono = mono.mean(axis=1, dtype=np.float32) / np.float32(32768.0)
            output = mono if stream is None else stream.resample_chunk(mono, last=False)
            return output

        def score(frame: array) -> float:
            nonlocal h, c, context
            audio = np.asarray(frame, dtype=np.float32).reshape(1, VAD_FRAME_SAMPLES)
            inputs = np.concatenate((context, audio), axis=1)
            probabilities, h, c = session.run(None, {"input": inputs, "h": h, "c": c})
            context = audio[:, -64:].copy()
            return float(np.asarray(probabilities).reshape(-1)[0])

        return PeerAudioStream(sample_rate, channels, resample, score, UtteranceSegmenter(config.vad, on_utterance))

    def transcribe(samples: array) -> dict[str, Any]:
        segments, info = model.transcribe(
            np.asarray(samples, dtype=np.float32),
            language=None,
            task="transcribe",
            beam_size=1,
            temperature=0.0,
            condition_on_previous_text=False,
            vad_filter=False,
            without_timestamps=True,
        )
        text_parts: list[str] = []
        text_length = 0
        for segment in segments:
            text_length += len(segment.text)
            if text_length > MAX_TRANSCRIPT_CHARS:
                raise RuntimeError("ASR transcript exceeded the output size limit")
            text_parts.append(segment.text)
        probability = float(info.language_probability)
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise RuntimeError("ASR returned an invalid language probability")
        # Empty text and languages other than zh/en still receive a terminal
        # transcript. The Node scheduler owns language/confidence policy.
        return {
            "text": "".join(text_parts).strip(),
            "language": info.language,
            "languageProbability": probability,
        }

    return peer_factory, transcribe


def consume_input(
    source: BinaryIO,
    handle: Callable[[Any], bool],
    emit: Callable[[dict[str, Any]], None],
    max_line_bytes: int = MAX_LINE_BYTES,
) -> None:
    """Read bounded records and recover at the next newline after invalid input."""
    def reject_constant(_value: str) -> None:
        raise ValueError("non-finite JSON numbers are not supported")

    while True:
        raw = source.readline(max_line_bytes + 1)
        if not raw:
            return
        if len(raw) > max_line_bytes:
            while raw and not raw.endswith(b"\n"):
                raw = source.readline(max_line_bytes + 1)
            emit({"type": "error", "message": "input JSON line exceeded the size limit"})
            continue
        try:
            message = json.loads(raw.decode("utf-8"), parse_constant=reject_constant)
        except (UnicodeError, ValueError, RecursionError):
            emit({"type": "error", "message": "invalid UTF-8 JSON command"})
            continue
        if not handle(message):
            return


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-json", required=True, help="Serialized local ASR configuration")
    arguments = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="strict", newline="\n", write_through=True)
        sys.stderr.reconfigure(encoding="utf-8", errors="replace", write_through=True)
    emit = JsonEmitter(sys.stdout)
    worker: TranslationWorker | None = None
    try:
        # Route dependency print statements away from the protocol, including
        # print statements executed by the inference thread.
        with redirect_stdout(sys.stderr):
            try:
                config = WorkerConfig.from_dict(json.loads(arguments.config_json))
                peer_factory, transcriber = load_runtime(config)
                worker = TranslationWorker(config, emit, peer_factory, transcriber)
                worker.start()
                emit({"type": "ready"})
                consume_input(sys.stdin.buffer, worker.handle, emit)
            finally:
                if worker is not None:
                    worker.close()
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        emit({"type": "error", "message": str(exc)[:2048], "fatal": True})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
