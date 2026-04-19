"""
voice_agent/agent/conversation.py

Main pipeline orchestrator.

Pipeline flow (VAD mode):
  Mic → VAD → utterance queue → STT → LLM stream → TTS chunks → Speaker

Key latency optimisations implemented here:
  1. STT runs immediately when VAD detects end of utterance
  2. LLM streams tokens
  3. TTS starts on the first sentence, not the full response
  4. Mic is muted during playback to prevent echo
  5. All heavy work done outside the audio callback thread
"""

import time
import queue
import threading
import numpy as np

from audio.capture  import MicCapture
from audio.vad      import VADProcessor
from audio.playback import play_audio
from stt.transcriber import Transcriber
from llm.responder   import LLMResponder
from tts.synthesizer import get_synthesizer
from config import INTERACTION_MODE, SAMPLE_RATE


class VoiceAgent:
    def __init__(self):
        print("\n[Agent] Initialising pipeline...")

        # Components
        self.mic         = MicCapture()
        self.transcriber = Transcriber()
        self.llm         = LLMResponder()
        self.tts         = get_synthesizer()

        # VAD feeds complete utterances into this queue
        self._utt_queue  = queue.Queue()
        self.vad         = VADProcessor(self._on_utterance)

        self._running    = False
        self._processing = False   # True while we are generating a response

        print("[Agent] Ready.\n")

    # ── VAD callback (called from capture thread) ─────────────────────────────
    def _on_utterance(self, audio: np.ndarray):
        """Called by VAD when a complete utterance is detected."""
        if self._processing:
            print("[Agent] Ignoring utterance (still processing previous)")
            return
        self._utt_queue.put(audio)

    # ── Main entry point ──────────────────────────────────────────────────────
    def run(self):
        if INTERACTION_MODE == "ptt":
            self._run_ptt()
        else:
            self._run_vad()

    # ── VAD mode ──────────────────────────────────────────────────────────────
    def _run_vad(self):
        print("[Agent] VAD mode – speak naturally. Ctrl+C to quit.\n")
        self.mic.start()
        self._running = True

        try:
            while self._running:
                # Feed mic frames to VAD
                frame = self.mic.read_frame()
                if frame is not None:
                    self.vad.process_frame(frame)

                # Process any completed utterances
                try:
                    audio = self._utt_queue.get_nowait()
                    self._handle_utterance(audio)
                except queue.Empty:
                    pass

                # Tiny sleep to avoid burning the CPU in the polling loop
                time.sleep(0.001)

        except KeyboardInterrupt:
            print("\n[Agent] Stopped.")
        finally:
            self.mic.stop()

    # ── Push-to-talk mode ─────────────────────────────────────────────────────
    def _run_ptt(self):
        print("[Agent] Push-to-talk mode.")
        print("  Press ENTER to start recording, then ENTER again to stop.\n")
        self.mic.start()
        self._running = True

        try:
            while self._running:
                input("  >> Press ENTER to speak...")
                print("  [Recording – press ENTER to stop]")
                self.mic.is_playing = False

                frames = []
                stop_event = threading.Event()

                def _record():
                    while not stop_event.is_set():
                        frame = self.mic.read_frame()
                        if frame:
                            frames.append(frame)
                        time.sleep(0.001)

                t = threading.Thread(target=_record, daemon=True)
                t.start()
                input()   # wait for second ENTER
                stop_event.set()
                t.join()

                if frames:
                    audio = np.frombuffer(b"".join(frames), dtype=np.int16)
                    self._handle_utterance(audio)

        except KeyboardInterrupt:
            print("\n[Agent] Stopped.")
        finally:
            self.mic.stop()

    # ── Core pipeline ─────────────────────────────────────────────────────────
    def _handle_utterance(self, audio: np.ndarray):
        self._processing = True
        t_start = time.time()

        try:
            # ── 1. STT ────────────────────────────────────────────────────────
            print("[Agent] Transcribing...")
            user_text = self.transcriber.transcribe(audio)

            if not user_text or len(user_text.strip()) < 2:
                print("[Agent] (empty or noise – skipping)")
                return

            print(f"\nYou: {user_text}")

            # ── 2. LLM + TTS streaming ────────────────────────────────────────
            # Mute mic during playback to prevent echo
            self.mic.is_playing = True
            self.mic.flush_queue()

            full_reply = ""
            first_chunk_played = False

            # Pre-synthesis pipeline: synthesize next chunk while current one plays
            import threading
            import queue as _queue

            audio_queue = _queue.Queue()  # holds (pcm_bytes, sr, text_chunk) tuples
            synthesis_done = threading.Event()

            def _synthesize_worker():
                """Background worker: pull text chunks from LLM, synthesize, enqueue audio."""
                for text_chunk in self.llm.generate_stream(user_text):
                    if not text_chunk:
                        continue
                    pcm_bytes, sr = self.tts.synthesize(text_chunk)
                    if pcm_bytes:
                        audio_queue.put((pcm_bytes, sr, text_chunk))
                synthesis_done.set()

            synth_thread = threading.Thread(target=_synthesize_worker, daemon=True)
            synth_thread.start()

            # Main thread: play audio as soon as it's ready
            while True:
                try:
                    pcm_bytes, sr, text_chunk = audio_queue.get(timeout=0.1)
                except _queue.Empty:
                    if synthesis_done.is_set() and audio_queue.empty():
                        break
                    continue

                if not first_chunk_played:
                    t_first = time.time() - t_start
                    print(f"[Agent] First audio in {t_first:.2f}s total latency")
                    first_chunk_played = True

                # Play audio (skip if TTS already played through system speakers)
                if not self.tts.plays_directly:
                    audio_chunk = np.frombuffer(pcm_bytes, dtype=np.int16)
                    play_audio(audio_chunk, sample_rate=sr)
                full_reply += text_chunk + " "

            synth_thread.join()

            print(f"\nAgent: {full_reply.strip()}")
            print(f"[Agent] Total response time: {time.time()-t_start:.2f}s\n")

        finally:
            # Unmute mic
            self.mic.is_playing = False
            self.mic.flush_queue()
            self._processing = False
