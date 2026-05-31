from __future__ import annotations

import re
from pathlib import Path


PREFERRED_CODE_FENCE_RE = re.compile(r"```(?:cpp|c\+\+|cxx|sycl)\s*\n?(.*?)```", re.S | re.I)
CODE_FENCE_RE = re.compile(r"```(?:[A-Za-z0-9_+\-]+)?\s*\n?(.*?)```", re.S | re.I)


def extract_code_block(text: str) -> str:
    if text is None:
        return ""
    match = PREFERRED_CODE_FENCE_RE.search(text) or CODE_FENCE_RE.search(text)
    if match is not None:
        return match.group(1).strip() + "\n"
    return text.strip() + "\n"


def normalize_sycl_source(text: str) -> str:
    code = extract_code_block(text)
    code = code.replace("```", "").strip() + "\n"
    if "#include" not in code:
        code = "#include <sycl/sycl.hpp>\nusing namespace sycl;\n\n" + code
    if "void gemm(" not in code:
        raise ValueError("Generated code does not contain the required `void gemm(...)` function.")
    if "sycl::queue" not in code and "queue &" not in code and "queue&" not in code:
        raise ValueError("Generated `gemm` must accept a SYCL queue reference.")
    return code


def save_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path
