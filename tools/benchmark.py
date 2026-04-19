"""
voice_agent/tools/benchmark.py

Measures per-stage latency without needing a microphone.
Generates a synthetic utterance and runs it through STT → LLM → TTS.

Run:
    python tools/benchmark.py
"""

import sys
import os
import time
import wave
import struct
import math
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import config


def generate_sine_audio(duration_s=2.0, freq=440, sr=16000) -> np.ndarray:
    """Generate a synthetic tone as int16 PCM (not real speech, just for timing)."""
    n = int(duration_s * sr)
    t = np.linspace(0, duration_s, n)
    wave_data = (np.sin(2 * math.pi * freq * t) * 16000).astype(np.int16)
    return wave_data


def make_test_audio_with_speech() -> np.ndarray:
    """
    If you have a pre-recorded WAV, load it here.
    Otherwise, returns a silent array (Whisper will return empty string).
    """
    test_wav = os.path.join(os.path.dirname(__file__), "test_speech.wav")
    if os.path.exists(test_wav):
        with wave.open(test_wav, "rb") as wf:
            frames = wf.readframes(wf.getnframes())
            return np.frombuffer(frames, dtype=np.int16)
    else:
        print("[Bench] No test_speech.wav found – using silence (STT will be empty)")
        return np.zeros(int(2.0 * config.SAMPLE_RATE), dtype=np.int16)


def run_benchmark():
    print("=" * 56)
    print("  Voice Agent – Latency Benchmark")
    print("=" * 56)

    results = {}

    # ── 1. STT ──────────────────────────────────────────────────────────────
    print("\n[1/3] STT benchmark (faster-whisper)...")
    from stt.transcriber import Transcriber
    trans = Transcriber()
    audio = make_test_audio_with_speech()

    # Warm up (first call is slower due to model initialisation)
    trans.transcribe(audio)

    times = []
    for i in range(3):
        t0 = time.time()
        text = trans.transcribe(audio)
        times.append(time.time() - t0)

    avg_stt = sum(times) / len(times)
    results["STT (avg 3 runs)"] = avg_stt
    print(f"  → avg {avg_stt:.3f}s  (text: '{text[:50]}')")

    # ── 2. LLM TTFT ─────────────────────────────────────────────────────────
    print("\n[2/3] LLM benchmark (Ollama – time to first token)...")
    from llm.responder import LLMResponder
    llm = LLMResponder()

    test_prompt = "What's two plus two?"
    t0 = time.time()
    first_token_time = None
    full_response = ""
    for chunk in llm.generate_stream(test_prompt):
        if first_token_time is None:
            first_token_time = time.time() - t0
        full_response += chunk
    total_llm = time.time() - t0

    results["LLM TTFT"]          = first_token_time or 0
    results["LLM total"]         = total_llm
    results["LLM response text"] = full_response.strip()[:60]
    print(f"  → TTFT: {first_token_time:.3f}s  |  total: {total_llm:.3f}s")
    print(f"  → Response: '{full_response.strip()[:60]}'")

    # ── 3. TTS ──────────────────────────────────────────────────────────────
    print("\n[3/3] TTS benchmark...")
    from tts.synthesizer import get_synthesizer
    tts = get_synthesizer()

    test_sentences = [
        "Hello, how can I help you today?",
        "The quick brown fox jumps over the lazy dog.",
    ]
    tts_times = []
    for sentence in test_sentences:
        t0 = time.time()
        pcm, sr = tts.synthesize(sentence)
        elapsed = time.time() - t0
        tts_times.append(elapsed)
        audio_ms = len(pcm) / 2 / sr * 1000 if pcm and sr else 0
        print(f"  → '{sentence[:40]}': synth={elapsed:.3f}s, audio={audio_ms:.0f}ms")

    avg_tts = sum(tts_times) / len(tts_times)
    results["TTS (avg)"] = avg_tts

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 56)
    print("  RESULTS SUMMARY")
    print("=" * 56)
    for key, val in results.items():
        if isinstance(val, float):
            print(f"  {key:<28} {val:.3f}s")
        else:
            print(f"  {key:<28} {val}")

    total_pipeline = results.get("STT (avg 3 runs)", 0) + \
                     results.get("LLM TTFT", 0) + \
                     results.get("TTS (avg)", 0)
    print(f"\n  ESTIMATED end-to-end latency:  {total_pipeline:.2f}s")
    print("  (VAD silence detection adds ~0.75s on top)")
    print("=" * 56)


if __name__ == "__main__":
    run_benchmark()
