"""Fake-model tests; explicitly run only after the user approves tests.

python -m unittest discover -s services/speech_translation/providers/faster_whisper -p test_worker.py
from the Node directory. No NumPy, ONNX, CUDA, model download, microphone, or
room connection is used. Importing worker itself loads standard library only.
"""

from array import array
import base64
from dataclasses import replace
import io
import json
import struct
import threading
import unittest

from worker import (
    JsonEmitter,
    PeerAudioStream,
    SAMPLE_RATE,
    TranslationWorker,
    UtteranceSegmenter,
    VadConfig,
    WorkerConfig,
    consume_input,
)


class EventCollector:
    def __init__(self):
        self.records = []
        self.condition = threading.Condition()

    def __call__(self, event):
        with self.condition:
            self.records.append(event)
            self.condition.notify_all()

    def of_type(self, event_type):
        with self.condition:
            return [record for record in self.records if record["type"] == event_type]

    def wait_for(self, predicate, timeout=2):
        with self.condition:
            if not self.condition.wait_for(lambda: any(predicate(item) for item in self.records), timeout):
                raise AssertionError("Timed out waiting for fake-worker event")
            return next(item for item in self.records if predicate(item))


def fake_peer_factory(config):
    def make_peer(sample_rate, channels, completed):
        if sample_rate != SAMPLE_RATE:
            raise ValueError("fake audio decoder uses 16 kHz")

        def decode(pcm):
            samples = struct.unpack("<" + "h" * (len(pcm) // 2), pcm)
            return [sum(samples[index : index + channels]) / channels / 32768.0
                    for index in range(0, len(samples), channels)]

        def score(frame):
            return 0.99 if max(abs(value) for value in frame) > 0.1 else 0.01

        return PeerAudioStream(sample_rate, channels, decode, score, UtteranceSegmenter(config.vad, completed))
    return make_peer


def audio_message(peer_uuid, value=12000, count=512, channels=1):
    return {
        "type": "audio", "peerUuid": peer_uuid, "sampleRate": SAMPLE_RATE,
        "channels": channels,
        "samples": base64.b64encode(struct.pack("<" + "h" * count, *([value] * count))).decode("ascii"),
    }


class SegmenterTests(unittest.TestCase):
    def test_pause_finishes_once_with_preroll_and_ignores_short_noise(self):
        completed = []
        segmenter = UtteranceSegmenter(VadConfig(), completed.append)
        for _ in range(12):
            segmenter.feed([0.0] * 512, 0.0)
        for _ in range(10):  # 320 ms of actual speech exceeds minSpeechMs.
            segmenter.feed([0.5] * 512, 1.0)
        for _ in range(21):
            segmenter.feed([0.0] * 512, 0.0)
        self.assertEqual(completed, [])
        segmenter.feed([0.0] * 512, 0.0)  # 704 ms silence ends the sentence.
        self.assertEqual(len(completed), 1)
        self.assertEqual(len(completed[0]), 4800 + 10 * 512 + 22 * 512)
        self.assertTrue(all(value == 0 for value in completed[0][:4800]))
        for _ in range(2):
            segmenter.feed([0.8] * 512, 1.0)
        for _ in range(23):
            segmenter.feed([0.0] * 512, 0.0)
        self.assertEqual(len(completed), 1)

    def test_hard_limit_is_exact_and_does_not_drop_frame_remainder(self):
        completed = []
        config = VadConfig(pre_roll_ms=0, max_speech_ms=1000)
        segmenter = UtteranceSegmenter(config, completed.append)
        for _ in range(32):
            segmenter.feed([0.5] * 512, 1.0)
        self.assertEqual(len(completed), 1)
        self.assertEqual(len(completed[0]), 16000)
        self.assertEqual(len(segmenter.active), 384)
        self.assertEqual(segmenter.speech_samples, 384)

    def test_preroll_and_partial_frame_storage_remain_bounded(self):
        completed = []
        segmenter = UtteranceSegmenter(VadConfig(), completed.append)
        stream = PeerAudioStream(SAMPLE_RATE, 1, lambda data: [0.0] * len(data), lambda _frame: 0.0, segmenter)
        for _ in range(1000):
            stream.feed(b"0" * 160)
            self.assertLess(len(stream.frame_buffer), 512)
        self.assertLessEqual(len(segmenter.pre_roll), 4800)
        self.assertIsNone(segmenter.active)
        self.assertEqual(completed, [])

    def test_packet_boundaries_do_not_change_vad_frames(self):
        observed = []
        segmenter = UtteranceSegmenter(VadConfig(), lambda _audio: None)
        stream = PeerAudioStream(
            SAMPLE_RATE, 1, lambda pcm: list(pcm),
            lambda frame: observed.append(list(frame)) or 0.0, segmenter,
        )
        stream.feed(bytes([1]) * 160)
        stream.feed(bytes([2]) * 160)
        stream.feed(bytes([3]) * 160)
        self.assertEqual(observed, [])
        stream.feed(bytes([4]) * 160)
        self.assertEqual(observed, [[1.0] * 160 + [2.0] * 160 + [3.0] * 160 + [4.0] * 32])
        self.assertEqual(list(stream.frame_buffer), [4.0] * 128)


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.config = WorkerConfig(model_path="unused-by-fake-model")
        self.events = EventCollector()
        self.calls = []
        self.worker = self.make_worker()

    def tearDown(self):
        self.worker.close()

    def make_worker(self, transcriber=None, config=None):
        config = config or self.config

        def transcribe(samples):
            self.calls.append(len(samples))
            return {"text": "你好", "language": "zh", "languageProbability": 0.95}

        worker = TranslationWorker(
            config, self.events, fake_peer_factory(config), transcriber or transcribe,
            clock_ms=lambda: 1234567890000,
        )
        worker.start()
        return worker

    def finish_sentence(self, peer="peer-a"):
        for _ in range(10):
            self.worker.handle(audio_message(peer))
        for _ in range(22):
            self.worker.handle(audio_message(peer, value=0))
        return self.events.of_type("utterance")[-1]

    def test_retention_requires_explicit_transcribe(self):
        event = self.finish_sentence()
        self.assertEqual(self.calls, [])
        self.assertEqual(event["endedAt"], 1234567890000)
        self.assertIn(event["requestId"], self.worker.pending)
        self.worker.handle({"type": "transcribe", "requestId": event["requestId"]})
        result = self.events.wait_for(lambda item: item["type"] == "transcript")
        self.assertEqual(result["text"], "你好")
        self.assertEqual(result["peerUuid"], "peer-a")
        self.assertEqual(self.calls, [10 * 512 + 22 * 512])
        self.assertNotIn(event["requestId"], self.worker.pending)

    def test_two_peers_never_share_vad_buffer(self):
        for _ in range(10):
            self.worker.handle(audio_message("speaker"))
            self.worker.handle(audio_message("silent", value=0))
        for _ in range(22):
            self.worker.handle(audio_message("speaker", value=0))
            self.worker.handle(audio_message("silent", value=0))
        events = self.events.of_type("utterance")
        self.assertEqual([event["peerUuid"] for event in events], ["speaker"])
        self.assertEqual(self.calls, [])

    def test_reset_drops_live_audio_but_keeps_completed_job(self):
        event = self.finish_sentence()
        self.worker.handle(audio_message("peer-a"))
        self.worker.handle({"type": "reset_peer", "peerUuid": "peer-a"})
        self.assertNotIn("peer-a", self.worker.peers)
        self.assertIn(event["requestId"], self.worker.pending)
        self.worker.handle({"type": "transcribe", "requestId": event["requestId"]})
        self.events.wait_for(lambda item: item["type"] == "transcript")

    def test_remove_discards_pending_and_live_state(self):
        event = self.finish_sentence()
        self.worker.handle({"type": "remove_peer", "peerUuid": "peer-a"})
        self.assertNotIn("peer-a", self.worker.peers)
        self.assertNotIn(event["requestId"], self.worker.pending)
        error = self.events.of_type("error")[-1]
        self.assertEqual(error["requestId"], event["requestId"])

    def test_discard_is_idempotent_and_frees_capacity(self):
        event = self.finish_sentence()
        for _ in range(2):
            self.worker.handle({"type": "discard", "requestId": event["requestId"]})
        self.assertEqual(self.worker.pending, {})
        self.assertEqual(self.events.of_type("error"), [])

    def test_pending_limit_drops_new_without_auto_inference(self):
        self.worker.close()
        self.worker = self.make_worker(config=replace(self.config, max_pending_utterances=1))
        first = self.finish_sentence()
        self.finish_sentence("peer-b")
        self.assertEqual(list(self.worker.pending), [first["requestId"]])
        self.assertEqual(len(self.events.of_type("utterance")), 1)
        self.assertIn("limit", self.events.of_type("error")[-1]["message"])
        self.assertEqual(self.calls, [])

    def test_reader_continues_and_cancel_replies_after_active_asr(self):
        self.worker.close()
        started = threading.Event()
        release = threading.Event()

        def delayed(_samples):
            started.set()
            if not release.wait(2):
                raise RuntimeError("fake inference was not released")
            return {"text": "hello", "language": "en", "languageProbability": 1.0}

        self.worker = self.make_worker(transcriber=delayed)
        first = self.finish_sentence()
        self.worker.handle({"type": "transcribe", "requestId": first["requestId"]})
        self.assertTrue(started.wait(2))
        try:
            second = self.finish_sentence("peer-b")
            self.assertIn(second["requestId"], self.worker.pending)
            self.worker.handle({"type": "reset_peer", "peerUuid": "peer-a"})
            self.assertFalse(self.worker.active.cancelled)
            self.worker.handle({"type": "remove_peer", "peerUuid": "peer-a"})
            self.assertTrue(self.worker.active.cancelled)
        finally:
            release.set()
        result = self.events.wait_for(lambda item: item["type"] == "error" and item.get("requestId") == first["requestId"])
        self.assertIn("canceled", result["message"])
        self.assertEqual(self.events.of_type("transcript"), [])
        self.worker.handle({"type": "transcribe", "requestId": second["requestId"]})
        self.events.wait_for(lambda item: item["type"] == "transcript" and item["requestId"] == second["requestId"])

    def test_empty_and_unsupported_language_still_terminal(self):
        self.worker.close()
        self.worker = self.make_worker(transcriber=lambda _samples: {"text": "", "language": "ja", "languageProbability": 0.8})
        event = self.finish_sentence()
        self.worker.handle({"type": "transcribe", "requestId": event["requestId"]})
        result = self.events.wait_for(lambda item: item["type"] == "transcript")
        self.assertEqual(result["text"], "")
        self.assertEqual(result["language"], "ja")

    def test_inference_failure_is_terminal_and_next_job_can_run(self):
        self.worker.close()
        count = 0

        def sometimes_fails(_samples):
            nonlocal count
            count += 1
            if count == 1:
                raise RuntimeError("fake CUDA error")
            return {"text": "ok", "language": "en", "languageProbability": 0.9}

        self.worker = self.make_worker(transcriber=sometimes_fails)
        first = self.finish_sentence()
        self.worker.handle({"type": "transcribe", "requestId": first["requestId"]})
        self.events.wait_for(lambda item: item["type"] == "error" and item.get("requestId") == first["requestId"])
        second = self.finish_sentence("peer-b")
        self.worker.handle({"type": "transcribe", "requestId": second["requestId"]})
        self.events.wait_for(lambda item: item["type"] == "transcript" and item["requestId"] == second["requestId"])

    def test_bad_audio_is_rejected_before_allocating_peer(self):
        invalid = [
            {**audio_message("invalid"), "samples": "not base64!"},
            {**audio_message("invalid"), "samples": base64.b64encode(b"x").decode()},
            {**audio_message("invalid"), "sampleRate": True},
            {**audio_message("invalid"), "channels": 3},
            audio_message("invalid", count=4001),
        ]
        for message in invalid:
            self.assertTrue(self.worker.handle(message))
        self.assertEqual(self.worker.peers, {})
        self.assertEqual(len(self.events.of_type("error")), len(invalid))

    def test_peer_limit_and_channel_change_rebuild_live_state(self):
        self.worker.close()
        self.worker = self.make_worker(config=replace(self.config, max_peers=1))
        completed = self.finish_sentence()
        self.worker.handle(audio_message("peer-a"))
        original_peer = self.worker.peers["peer-a"]
        self.worker.handle(audio_message("peer-b"))
        self.assertEqual(set(self.worker.peers), {"peer-a"})
        self.worker.handle(audio_message("peer-a", channels=2))
        self.assertEqual(self.worker.peers["peer-a"].channels, 2)
        self.assertIsNot(self.worker.peers["peer-a"], original_peer)
        self.assertEqual(len(self.worker.peers["peer-a"].frame_buffer), 256)
        self.assertIsNone(self.worker.peers["peer-a"].segmenter.active)
        self.assertIn(completed["requestId"], self.worker.pending)
        self.assertEqual(len(self.events.of_type("error")), 1)

    def test_sample_rate_change_preserves_active_and_pending_jobs(self):
        self.worker.close()
        started = threading.Event()
        release = threading.Event()

        def delayed(_samples):
            started.set()
            if not release.wait(2):
                raise RuntimeError("fake inference was not released")
            return {"text": "hello", "language": "en", "languageProbability": 1.0}

        self.worker = self.make_worker(transcriber=delayed)
        active = self.finish_sentence()
        pending = self.finish_sentence()
        original_peer = self.worker.peers["peer-a"]
        self.worker.handle({"type": "transcribe", "requestId": active["requestId"]})
        self.assertTrue(started.wait(2))
        calls = []

        def new_format_peer(sample_rate, channels, completed):
            calls.append((sample_rate, channels))
            # This test exercises format handling; native resampling is not loaded.
            return PeerAudioStream(sample_rate, channels, lambda _pcm: [], lambda _frame: 0,
                                   UtteranceSegmenter(self.config.vad, completed))

        self.worker.peer_factory = new_format_peer
        try:
            self.worker.handle({**audio_message("peer-a"), "sampleRate": 48000})
            self.assertEqual(calls, [(48000, 1)])
            self.assertIsNot(self.worker.peers["peer-a"], original_peer)
            self.assertFalse(self.worker.active.cancelled)
            self.assertIn(pending["requestId"], self.worker.pending)
            self.assertEqual(self.events.of_type("error"), [])
        finally:
            release.set()
        self.events.wait_for(lambda item: item["type"] == "transcript" and item["requestId"] == active["requestId"])


class ProtocolTests(unittest.TestCase):
    def test_invalid_and_oversized_lines_recover_at_newline(self):
        received = []
        errors = []
        source = io.BytesIO(b"x" * 200 + b"\n" + b"\xff\n" + b'{"type":"shutdown"}\n')
        consume_input(source, lambda message: received.append(message) or False, errors.append, max_line_bytes=64)
        self.assertEqual(received, [{"type": "shutdown"}])
        self.assertEqual(len(errors), 2)

    def test_utf8_json_output_has_no_status_text_or_split_lines(self):
        destination = io.StringIO()
        emit = JsonEmitter(destination)
        emit({"type": "transcript", "text": "中文\nEnglish"})
        lines = destination.getvalue().splitlines()
        self.assertEqual(len(lines), 1)
        self.assertEqual(json.loads(lines[0])["text"], "中文\nEnglish")

    def test_config_rejects_cpu_and_excessive_retained_audio(self):
        with self.assertRaises(ValueError):
            WorkerConfig.from_dict({"modelPath": "local", "device": "cpu"})
        with self.assertRaises(ValueError):
            WorkerConfig.from_dict({"modelPath": "local", "maxPendingUtterances": 1000})
        with self.assertRaises(ValueError):
            WorkerConfig.from_dict({"modelPath": "local", "vad": {"maxSpeechMs": 20000}})


if __name__ == "__main__":
    unittest.main()
