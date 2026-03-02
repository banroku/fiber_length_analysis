# src/fiberlen/save_fiber_list_csv.py

from __future__ import annotations

import csv
import io
from typing import Any, Sequence


def save_fiber_list_csv(fibers: Sequence[Any], filename: str, um_per_px: float) -> str:
    """
    fibers を列挙するだけの CSV を作り、CSV文字列を返す。

    出力列: fiber_id, length_um, percentile
      - length_um = length_px * um_per_px
      - percentile = length-weighted累積分率（0?1）, 小数点3桁

    備考:
      - filename 引数はシグネチャ維持のため残す（この版では未使用）
      - fibers は昇順ソート前提だが、安全のため length_px で再ソートする
      - fiber は .fiber_id と .length_px を持つ前提
    """
    #_ = filename  # 未使用（将来の拡張用）

    fibers_sorted = sorted(fibers, key=lambda f: float(f.length_px))

    lengths_um = [float(f.length_px) * float(um_per_px) for f in fibers_sorted]
    total_len = float(sum(lengths_um))

    # length-weighted cumulative fraction
    if total_len > 0.0:
        cum = 0.0
        percentiles = []
        for L in lengths_um:
            cum += float(L)
            percentiles.append(cum / total_len)
    else:
        percentiles = [0.0 for _ in lengths_um]

    sio = io.StringIO()
    w = csv.writer(sio, lineterminator="\n")

    # ヘッダ行（列名）
    w.writerow(["fiber_id", "length_um", filename])

    for f, L_um, frac in zip(fibers_sorted, lengths_um, percentiles):
        w.writerow([str(int(f.fiber_id)), f"{L_um:.1f}", f"{frac:.3f}"])

    return sio.getvalue()
