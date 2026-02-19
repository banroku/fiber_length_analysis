# Path: src/fiberlen/postprocess.py

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import csv
import json

import numpy as np

from fiberlen.types import Fiber


def postprocess(
    fiber: List[Fiber],
    eliminate_length: float,
    out_csv_path: str | Path,
    out_config_json_path: str | Path | None = None,
    used_config: Dict[str, Any] | None = None,
) -> List[Fiber]:
    """
    結果の後処理と保存。

    仕様（あなたの設計どおり）
    ------------------------
    ・短い繊維（length_px < eliminate_length）を除外
    ・Fiber は長い順にソート
    ・csv を出力（Excelで読める素直な形式）
    ・（任意）最終的に使った設定を json に保存

    Parameters
    ----------
    fiber : List[Fiber]
        measure_length の出力
    eliminate_length : float
        px 単位（内部単位は px で統一）
    out_csv_path : str | Path
        出力先（例: data/output/fibers.csv）
    out_config_json_path : str | Path | None
        出力先（例: data/output/used_config.json）
    used_config : Dict[str,Any] | None
        設定値の辞書（pipeline側でまとめて渡す想定）

    Returns
    -------
    filtered_sorted : List[Fiber]
        フィルタ後、length_px の降順
    """
    thr = float(eliminate_length)

    filtered = [f for f in fiber if float(getattr(f, "length_px", 0.0)) >= thr]
    filtered.sort(key=lambda x: float(getattr(x, "length_px", 0.0)), reverse=True)

    out_csv_path = Path(out_csv_path)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)

    # csv: fiber_id, length_px, n_segments, seg_ids
    with out_csv_path.open("w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow(["fiber_id", "length_px", "n_segments", "seg_ids"])
        for f in filtered:
            seg_ids = list(getattr(f, "seg_ids", []))
            w.writerow(
                [
                    int(getattr(f, "fiber_id", -1)),
                    float(getattr(f, "length_px", 0.0)),
                    int(len(seg_ids)),
                    " ".join(str(s) for s in seg_ids),
                ]
            )

    if out_config_json_path is not None:
        out_config_json_path = Path(out_config_json_path)
        out_config_json_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "eliminate_length_px": thr,
            "n_fibers_in": int(len(fiber)),
            "n_fibers_out": int(len(filtered)),
            "used_config": used_config or {},
        }
        with out_config_json_path.open("w", encoding="utf-8") as fp:
            json.dump(payload, fp, ensure_ascii=False, indent=2)

    return filtered
