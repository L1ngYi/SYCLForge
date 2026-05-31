from __future__ import annotations

import re
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


GEMM_STEM_RE = re.compile(r"^gemm_(\d+)_(\d+)_(\d+)$")


@dataclass(frozen=True)
class GemmTask:
    path: Path
    stem: str
    m: int
    k: int
    n: int

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.m, self.k, self.n

    @property
    def flops(self) -> float:
        return float(2 * self.m * self.k * self.n)

    @property
    def seed(self) -> int:
        return zlib.crc32(self.stem.encode("utf-8")) & 0xFFFFFFFF


def parse_gemm_task(path: Path) -> GemmTask:
    match = GEMM_STEM_RE.match(path.stem)
    if match is None:
        raise ValueError(f"Not a GEMM shape file: {path.name}")
    m_dim, k_dim, n_dim = (int(token) for token in match.groups())
    return GemmTask(path=path.resolve(), stem=path.stem, m=m_dim, k=k_dim, n=n_dim)


def _iter_candidate_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    if not path.is_dir():
        raise FileNotFoundError(path)
    yield from sorted(path.glob("gemm_*.cpp"))


def discover_tasks(
    path: Path,
    *,
    case_stem: str = "",
    first_n: int = 0,
) -> list[GemmTask]:
    tasks: list[GemmTask] = []
    wanted = case_stem.removesuffix(".cpp")
    for file_path in _iter_candidate_files(path):
        if wanted and file_path.stem != wanted:
            continue
        try:
            tasks.append(parse_gemm_task(file_path))
        except ValueError:
            continue

    if first_n > 0:
        tasks = tasks[:first_n]
    if not tasks:
        detail = f" matching {case_stem!r}" if case_stem else ""
        raise FileNotFoundError(f"No gemm_M_K_N.cpp tasks found in {path}{detail}")
    return tasks
