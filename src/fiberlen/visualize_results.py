# src/fiberlen/visualize_results.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Tuple, List, Dict, Optional
import time

import numpy as np
import matplotlib.pyplot as plt

from fiberlen.config import CFG


_OUTPUT_DIR: Optional[Path] = None
_TAG: Optional[str] = None


def configure_visualize_output(out_dir: str | Path, tag: str) -> None:
    global _OUTPUT_DIR, _TAG
    _OUTPUT_DIR = Path(out_dir)
    _TAG = str(tag)


def visualize_results(fiber: Any, hist_range: Tuple[float, float], hist_bins: int) -> None:
    """
    fiber: pipeline の最終結果（通常は list[Fiber]）
    hist_range: (min_um, max_um) 例: (0, 1500)
    hist_bins: ヒストグラムの区間数 例: 30

    統計はすべて「長さ重みづけ（weights=length）」で計算する。
    """
    out_dir = _OUTPUT_DIR if _OUTPUT_DIR is not None else (Path.cwd() / "data" / "output")
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = _TAG if _TAG is not None else time.strftime("results_%Y%m%d_%H%M%S")

    lengths_um_all = _extract_lengths_um(fiber, um_per_px=float(CFG.um_per_px))

    r0, r1 = float(hist_range[0]), float(hist_range[1])
    mask = (lengths_um_all >= r0) & (lengths_um_all <= r1)
    lengths_um = lengths_um_all[mask]

    # 繊維総数（“総数”は本数として出す。統計は長さ重みづけ）
    n_total = int(lengths_um_all.size)
    n_in_range = int(lengths_um.size)

    if n_in_range == 0:
        _write_text(
            out_dir / f"{tag}__summary.txt",
            "\n".join(
                [
                    f"tag: {tag}",
                    f"range_um: {r0} - {r1}",
                    f"bins: {int(hist_bins)}",
                    f"fiber_count_total: {n_total}",
                    f"fiber_count_in_range: {n_in_range}",
                    "note: no fibers in specified range",
                ]
            )
            + "\n",
        )
        return

    # 長さ重みづけ（各繊維の重み = その繊維の長さ[um]）
    weights = lengths_um.copy()

    # 平均繊維長（長さ重みづけ）
    # w = L なので mean = sum(L*w)/sum(w) = sum(L^2)/sum(L)
    mean_um = float(np.sum(lengths_um * weights) / np.sum(weights))

    # ヒストグラム（長さ重みづけ: 各binは総延長[um]）
    hist, edges = np.histogram(
        lengths_um,
        bins=int(hist_bins),
        range=(r0, r1),
        weights=weights,
    )

    # R表（長さ重みづけ分位点）
    percentiles = [0, 5, 10, 20, 50, 80, 90, 95, 100]
    r_table: Dict[str, float] = {
        f"R{p}": float(_weighted_quantile(lengths_um, weights, p / 100.0)) for p in percentiles
    }

    # summary txt
    total_length_um_in_range = float(np.sum(weights))
    _write_text(
        out_dir / f"{tag}__summary.txt",
        "\n".join(
            [
                f"tag: {tag}",
                f"range_um: {r0} - {r1}",
                f"bins: {int(hist_bins)}",
                f"fiber_count_total: {n_total}",
                f"fiber_count_in_range: {n_in_range}",
                f"mean_length_um (length-weighted): {mean_um}",
                f"total_length_um (in range): {total_length_um_in_range}",
            ]
        )
        + "\n",
    )

    # R table csv
    _write_text(
        out_dir / f"{tag}__R_table_length_weighted.csv",
        "metric,value_um\n" + "\n".join([f"{k},{v}" for k, v in r_table.items()]) + "\n",
    )

    # histogram plot png（色指定なし）
    fig = plt.figure()
    ax = fig.add_subplot(111)
    centers = (edges[:-1] + edges[1:]) / 2.0
    widths = np.diff(edges)
    ax.bar(centers, hist, width=widths, align="center")
    ax.set_xlabel("Fiber length (um)")
    ax.set_ylabel("Total length per bin (um) (length-weighted)")
    ax.set_title("Length-weighted histogram")
    fig.tight_layout()
    fig.savefig(out_dir / f"{tag}__hist_length_weighted.png", dpi=150)
    plt.close(fig)


def _extract_lengths_um(fiber: Any, *, um_per_px: float) -> np.ndarray:
    """
    fiber から length_px を取り出して μm に変換する。
    想定: list[types.Fiber]（length_px: float） :contentReference[oaicite:2]{index=2}
    互換: list で各要素が length_px 属性または length_px キーを持つ dict/obj
    """
    if isinstance(fiber, dict) and "fibers" in fiber:
        fiber = fiber["fibers"]

    if not isinstance(fiber, (list, tuple)):
        raise TypeError("fiber must be a list/tuple (or dict with key 'fibers').")

    out: List[float] = []
    for f in fiber:
        if f is None:
            continue

        if isinstance(f, dict):
            if "length_px" not in f:
                raise ValueError("fiber dict element must have 'length_px'.")
            lp = f["length_px"]
        else:
            if not hasattr(f, "length_px"):
                raise ValueError("fiber element must have attribute 'length_px'.")
            lp = getattr(f, "length_px")

        if lp is None:
            continue

        out.append(float(lp) * float(um_per_px))

    return np.asarray(out, dtype=float)


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """
    長さ重みづけ分位点。q in [0,1]
    """
    if not (0.0 <= q <= 1.0):
        raise ValueError("q must be in [0,1]")

    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    if v.size == 0:
        return float("nan")

    idx = np.argsort(v)
    v_sorted = v[idx]
    w_sorted = w[idx]

    cum_w = np.cumsum(w_sorted)
    total = float(cum_w[-1])
    if total <= 0.0:
        return float("nan")

    target = q * total
    pos = int(np.searchsorted(cum_w, target, side="left"))
    pos = max(0, min(pos, v_sorted.size - 1))
    return float(v_sorted[pos])


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
