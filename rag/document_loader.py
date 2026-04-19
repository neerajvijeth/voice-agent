"""
voice_agent/rag/document_loader.py

Parse .docx and .pptx files into text chunks with source metadata.
Each chunk carries its source filename and location (slide number or paragraph range)
so the agent can tell the user exactly where the information comes from.
"""

import os
from dataclasses import dataclass


@dataclass
class Chunk:
    """A text chunk with source metadata."""
    text: str
    source_file: str
    location: str       # e.g. "Slide 3" or "Paragraphs 5-10"


def load_docx(path: str) -> list[Chunk]:
    """Extract text from a .docx file, chunked by paragraph groups."""
    from docx import Document
    doc = Document(path)
    fname = os.path.basename(path)

    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    if not paragraphs:
        return []

    chunks = []
    buf = []
    buf_start = 1

    for i, para in enumerate(paragraphs, 1):
        buf.append(para)
        joined = " ".join(buf)
        if len(joined) >= 250 or i == len(paragraphs):
            location = f"Paragraph {buf_start}" if buf_start == i else f"Paragraphs {buf_start}-{i}"
            chunks.append(Chunk(text=joined, source_file=fname, location=location))
            buf = []
            buf_start = i + 1

    return chunks


def load_pptx(path: str) -> list[Chunk]:
    """Extract text from a .pptx file, one chunk per slide."""
    from pptx import Presentation
    prs = Presentation(path)
    fname = os.path.basename(path)
    chunks = []

    for slide_num, slide in enumerate(prs.slides, 1):
        texts = []
        for shape in slide.shapes:
            if shape.has_text_frame:
                for paragraph in shape.text_frame.paragraphs:
                    line = paragraph.text.strip()
                    if line:
                        texts.append(line)

        if texts:
            slide_text = " ".join(texts)
            chunks.append(Chunk(
                text=slide_text,
                source_file=fname,
                location=f"Slide {slide_num}",
            ))

    return chunks


def load_pdf(path: str) -> list[Chunk]:
    """Extract text from a PDF file, chunked by page."""
    from PyPDF2 import PdfReader
    reader = PdfReader(path)
    fname = os.path.basename(path)
    chunks = []

    for page_num, page in enumerate(reader.pages, 1):
        text = page.extract_text()
        if text and text.strip():
            chunks.append(Chunk(
                text=text.strip(),
                source_file=fname,
                location=f"Page {page_num}",
            ))

    return chunks


def load_txt(path: str) -> list[Chunk]:
    """Load a plain text file, chunked by ~300 char blocks."""
    fname = os.path.basename(path)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read().strip()

    if not content:
        return []

    # Split into ~300 char chunks on paragraph boundaries
    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
    chunks = []
    buf = []
    buf_start = 1

    for i, para in enumerate(paragraphs, 1):
        buf.append(para)
        joined = " ".join(buf)
        if len(joined) >= 300 or i == len(paragraphs):
            location = f"Section {buf_start}" if buf_start == i else f"Sections {buf_start}-{i}"
            chunks.append(Chunk(text=joined, source_file=fname, location=location))
            buf = []
            buf_start = i + 1

    return chunks


def load_directory(directory: str) -> list[Chunk]:
    """Load all supported documents from a directory."""
    all_chunks = []

    if not os.path.isdir(directory):
        print(f"[RAG] Documents directory not found: {directory}")
        return []

    for fname in sorted(os.listdir(directory)):
        fpath = os.path.join(directory, fname)
        if not os.path.isfile(fpath):
            continue

        try:
            if fname.endswith(".docx"):
                chunks = load_docx(fpath)
            elif fname.endswith(".pptx"):
                chunks = load_pptx(fpath)
            elif fname.endswith(".pdf"):
                chunks = load_pdf(fpath)
            elif fname.endswith(".txt"):
                chunks = load_txt(fpath)
            else:
                continue

            print(f"[RAG] Loaded {fname}: {len(chunks)} chunks")
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"[RAG] Error loading {fname}: {e}")

    print(f"[RAG] Total: {len(all_chunks)} chunks from {directory}")
    return all_chunks
