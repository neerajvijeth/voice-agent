"""
voice_agent/main.py

Entry point for the local voice agent.

Run:
    python main.py

Optional flags:
    python main.py --mode ptt     # push-to-talk instead of VAD
    python main.py --model tiny   # use faster (smaller) Whisper model
    python main.py --tts edge     # override TTS backend

Requires a .env file with GEMINI_API_KEY=your-key
"""

import sys
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Local low-latency voice agent")
    parser.add_argument("--mode",  default=None, choices=["vad", "ptt"],
                        help="Interaction mode (default from config.py)")
    parser.add_argument("--model", default=None,
                        help="Whisper model size (tiny/base/small)")
    parser.add_argument("--tts",   default=None, choices=["piper","edge","pyttsx3"],
                        help="TTS backend override")
    return parser.parse_args()


def apply_overrides(args):
    import config
    if args.mode:
        config.INTERACTION_MODE = args.mode
    if args.model:
        config.WHISPER_MODEL = args.model
    if args.tts:
        config.TTS_BACKEND = args.tts


def print_banner():
    print("""
╔══════════════════════════════════════════════════╗
║        LOCAL VOICE AGENT  –  MVP v1.0            ║
║                                                  ║
║  Stack: faster-whisper | Gemini | edge-tts/piper ║
║  Mode : see config.py INTERACTION_MODE           ║
║  Stop : Ctrl+C                                   ║
╚══════════════════════════════════════════════════╝
""")


def main():
    # Load .env file so GEMINI_API_KEY is available
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        print("[WARN] python-dotenv not installed. Set GEMINI_API_KEY as env var manually.")

    args = parse_args()
    apply_overrides(args)

    print_banner()

    from agent.conversation import VoiceAgent
    agent = VoiceAgent()
    agent.run()


if __name__ == "__main__":
    main()
