"""
voice_agent/llm/responder.py

LLM response generation via the Google Gemini API (streaming).

Why Gemini?
  - Much smarter than small local models (3B params) → better conversation quality
  - Extremely fast TTFT via Google's infrastructure
  - Streaming support → TTS starts on first sentence, not full response
  - Frees up local CPU/GPU for STT and TTS

Strategy for low latency:
  - Stream tokens from Gemini
  - Accumulate until we have a "speakable chunk" (sentence boundary or N tokens)
  - Yield the chunk for TTS to start immediately
  - Continue accumulating while first chunk plays
"""

import time
from google import genai
from config import (
    GEMINI_API_KEY, GEMINI_MODEL, LLM_MAX_TOKENS,
    LLM_TEMPERATURE, SYSTEM_PROMPT,
    RAG_ENABLED, RAG_DOCUMENTS_DIR, RAG_TOP_K,
)


class LLMResponder:
    def __init__(self):
        self.history = []   # list of {"role": ..., "content": ...}
        self._setup_client()
        self._setup_rag()

    def _setup_rag(self):
        """Initialise the RAG retriever if enabled."""
        self.retriever = None
        if RAG_ENABLED:
            try:
                from rag.retriever import RAGRetriever
                self.retriever = RAGRetriever(RAG_DOCUMENTS_DIR)
            except Exception as e:
                print(f"[RAG] Failed to initialise: {e}")
                self.retriever = None

    def _setup_client(self):
        """Initialise the Gemini client."""
        if not GEMINI_API_KEY:
            print("[LLM] ERROR: GEMINI_API_KEY not set. Add it to your .env file.")
            self.client = None
            return

        self.client = genai.Client(api_key=GEMINI_API_KEY)
        print(f"[LLM] Gemini ready (model: {GEMINI_MODEL})")

    def add_user_message(self, text: str):
        self.history.append({"role": "user", "content": text})

    def add_assistant_message(self, text: str):
        self.history.append({"role": "assistant", "content": text})

    def clear_history(self):
        self.history = []

    def _build_contents(self, rag_context: str = "") -> list:
        """
        Build the contents list for the Gemini API from conversation history.
        Maps our history format to Gemini's expected format.
        If rag_context is provided, it's prepended to the last user message.
        """
        contents = []
        for i, msg in enumerate(self.history):
            role = "user" if msg["role"] == "user" else "model"
            text = msg["content"]
            # Inject RAG context into the latest user message
            if rag_context and i == len(self.history) - 1 and role == "user":
                text = f"[Retrieved document context]\n{rag_context}\n\n[User question]\n{text}"
            contents.append({"role": role, "parts": [{"text": text}]})
        return contents

    def generate_stream(self, user_text: str):
        """
        Generator that streams speakable text chunks.
        Yields strings that should be passed to TTS one at a time.
        """
        if not self.client:
            yield "Sorry, the Gemini API key is not configured."
            return

        self.add_user_message(user_text)

        # ── RAG: retrieve relevant context ──
        rag_context = ""
        if self.retriever:
            context_str, sources = self.retriever.retrieve(user_text, top_k=RAG_TOP_K)
            if sources:
                rag_context = context_str
                print(f"[RAG] Sources:")
                for s in sources:
                    print(f"  → {s['file']}, {s['location']}: {s['snippet']}")

        contents = self._build_contents(rag_context=rag_context)

        t0            = time.time()
        first_token   = True
        token_buf     = ""   # accumulates until we have a speakable chunk
        full_response = ""

        try:
            response = self.client.models.generate_content_stream(
                model=GEMINI_MODEL,
                contents=contents,
                config={
                    "system_instruction": SYSTEM_PROMPT,
                    "max_output_tokens": LLM_MAX_TOKENS,
                    "temperature": LLM_TEMPERATURE,
                },
            )

            for chunk in response:
                token = chunk.text or ""
                if token:
                    if first_token:
                        print(f"[LLM] First token in {time.time()-t0:.2f}s")
                        first_token = False

                    token_buf     += token
                    full_response += token

                    # Yield when we hit a natural speech boundary
                    speakable = _extract_speakable_chunk(token_buf)
                    if speakable:
                        token_buf = token_buf[len(speakable):]
                        print(f"[LLM] chunk → TTS: '{speakable.strip()}'")
                        yield speakable.strip()

        except Exception as e:
            error_msg = "Sorry, I can't reach the language model right now."
            print(f"[LLM] Error: {e}")
            yield error_msg
            full_response = error_msg

        # Flush any remaining text
        if token_buf.strip():
            print(f"[LLM] final chunk → TTS: '{token_buf.strip()}'")
            yield token_buf.strip()

        self.add_assistant_message(full_response.strip())
        print(f"[LLM] Total: '{full_response.strip()}' ({time.time()-t0:.2f}s)")

    def generate_sync(self, user_text: str) -> str:
        """Non-streaming fallback. Returns complete response string."""
        return " ".join(self.generate_stream(user_text))


# ── Helper: chunk splitter ────────────────────────────────────────────────────

_SENTENCE_ENDS = {'.', '!', '?', '…'}
_CLAUSE_ENDS   = {',', ';', ':'}


def _extract_speakable_chunk(text: str) -> str | None:
    """
    Look for a good place to cut the text for TTS.
    Returns the chunk (including the delimiter) or None if not ready yet.

    Priority:
      1. Sentence end (. ! ?)  – always cut here
      2. Clause end (, ; :)    – cut if buffer is getting long
      3. Word boundary         – cut if buffer exceeds ~60 chars (fallback)
    """
    # Walk the string looking for natural breaks
    for i, ch in enumerate(text):
        if ch in _SENTENCE_ENDS:
            # Make sure we're not inside "3.14" or "U.S.A"
            if i + 1 < len(text) and text[i + 1].isdigit():
                continue
            return text[:i + 1]

    for i, ch in enumerate(text):
        if ch in _CLAUSE_ENDS and len(text) > 300: # practically disabled
            return text[:i + 1]

    # Hard cut on word boundary if buffer is very long
    if len(text) > 70:
        last_space = text.rfind(" ", 0, 65)
        if last_space > 0:
            return text[:last_space + 1]

    return None  # not enough text yet
