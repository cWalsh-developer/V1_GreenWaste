from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


def load_ikea_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"IKEA dataset not found: {path}")
    return pd.read_excel(path)


def read_docx_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"DOCX file not found: {path}")

    try:
        from docx import Document
    except ImportError as exc:
        message = (
            "python-docx is required to read .docx files. "
            "Install with: pip install python-docx"
        )
        raise ImportError(message) from exc

    document = Document(path)
    paragraphs = [p.text.strip() for p in document.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)
