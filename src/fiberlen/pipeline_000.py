# Path: src/fiberlen/pipeline.py
#
# 入出力フォルダ（project root ??）
#   data/input          : 生画像
#   data/intermediate   : 中間生成物（任意）
#   data/output         : 最終結果（csv, json, overlay等）
#
# 注意:
# ・関数名はユーザー仕様どおり固定
# ・内部単位は px
# ・設定値は config.py の CFG から供給

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import json
import numpy as np
import imageio.v3 as iio

from fiberlen.config import CFG

from fiberlen.input_img import input_img
from fiberlen.subtract_background import subtract_background
from fiberlen.calc_otsu_threshold import calc_Otsu_threshold
from fiberlen.binarize import binarize
from fiberlen.noize_elimination import noize_elimination
from fiberlen.skeletonize import skeletonize
from fiberlen.convert_to_graph import convert_to_graph
from fiberlen.kink_cut import kink_cut
from fiberlen.pairing import pairing
from fiberlen.measure_length import measure_length
from fiberlen.postprocess import postprocess
from fiberlen.types import CompressedGraph, Fiber


def run_pipeline(
    img_path: str | Path,
    *,
    save_intermediate: bool = True,
    out_tag: Optional[str] = None,
) -> Dict[str, Any]:
    """
    指定画像1枚を、あなたの設計順で一気通貫に処理する。

    Returns には、最終fiberと、推奨Otsu閾値、出力ファイルパスなどを入れる。
    """
    img_path = Path(img_path)

    project_root = _find_project_root(img_path)
    data_dir = project_root / "data"
    input_dir = data_dir / "input"
    inter_dir = data_dir / "intermediate"
    out_dir = data_dir / "output"

    inter_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = img_path.stem
    tag = out_tag.strip() if isinstance(out_tag, str) and out_tag.strip() else stem

    # ---- 1) 画像の読み込み ----
    img_raw01 = input_img(img_path, CFG.foreground_is_dark)  # float [0..1], 以降は背景黒前提

    # ---- 2) 背景均一化 ----
    img_preprocessed01 = subtract_background(img_raw01, CFG.blur_sigma_px)

    # ---- 3) Otsu推奨閾値（パイプラインには反映しない） ----
    threshold_otsu = float(calc_Otsu_threshold(img_preprocessed01))

    # ---- 4) 二値化 ----
    #img_binarized = binarize(img_preprocessed01, CFG.threshold)  # 0/1 あるいは bool 想定
    img_binarized = binarize(img_preprocessed01, threshold_otsu)  # 0/1 あるいは bool 想定

    # ---- 5) ノイズ除去 ----
    img_for_skeletonized = noize_elimination(img_binarized, CFG.eliminate_length_px)

    # ---- 6) スケルトナイズ ----
    img_skeletonized = skeletonize(img_for_skeletonized)

    # ---- 7) グラフ化 ----
    graph = convert_to_graph(img_skeletonized, CFG.border_margin_px)

    # ---- 8) キンク処理 ----
    graph_kink_cut = kink_cut(
        graph,
        CFG.threshold_of_nonlinear,
        CFG.blob_px,
        CFG.cut_max,
        CFG.cut_angle,
    )

    # ---- 9) pairing ----
    graph_paired = pairing(
        graph_kink_cut,
        CFG.pairing_angle_max,
        CFG.pairing_length_for_calc_angle,
    )

    # ---- 10) 長さ測定 ----
    fibers = measure_length(graph_paired, CFG.top_cut)

    # ---- 11) postprocess（csv保存）----
    fibers_csv = out_dir / f"{tag}__fibers.csv"
    used_cfg_json = out_dir / f"{tag}__used_config.json"
    fibers_filtered = postprocess(
        fibers,
        CFG.post_eliminate_length_px,
        out_csv_path=fibers_csv,
        out_config_json_path=used_cfg_json,
        used_config=_cfg_payload(threshold_otsu=threshold_otsu),
    )

    # ---- optional: intermediate 保存 ----
    saved = {}
    if save_intermediate:
        saved.update(
            _save_intermediate_images(
                inter_dir=inter_dir,
                tag=tag,
                img_raw01=img_raw01,
                img_preprocessed01=img_preprocessed01,
                img_binarized=img_binarized,
                img_skeletonized=img_skeletonized,
            )
        )
        # graphの保存（簡易json）
        gpath = inter_dir / f"{tag}__graph.json"
        _save_graph_json(graph_paired, gpath)
        saved["graph_json"] = str(gpath)

    return {
        "img_path": str(img_path),
        "tag": tag,
        "threshold_otsu": threshold_otsu,
        "threshold_used": float(CFG.threshold),
        "n_fibers_raw": int(len(fibers)),
        "n_fibers_final": int(len(fibers_filtered)),
        "fibers_csv": str(fibers_csv),
        "used_config_json": str(used_cfg_json),
        "saved_intermediate": saved,
    }


def _cfg_payload(*, threshold_otsu: float) -> Dict[str, Any]:
    d = asdict(CFG)
    d["threshold_otsu_recommended"] = float(threshold_otsu)
    return d


def _save_intermediate_images(
    *,
    inter_dir: Path,
    tag: str,
    img_raw01: np.ndarray,
    img_preprocessed01: np.ndarray,
    img_binarized: np.ndarray,
    img_skeletonized: np.ndarray,
) -> Dict[str, str]:
    out: Dict[str, str] = {}

    # raw / preprocessed は 0..1 float を 8bit で保存
    raw8 = np.clip(img_raw01 * 255.0, 0, 255).astype(np.uint8)
    pre8 = np.clip(img_preprocessed01 * 255.0, 0, 255).astype(np.uint8)

    p_raw = inter_dir / f"{tag}__raw.tif"
    p_pre = inter_dir / f"{tag}__preprocessed.tif"
    iio.imwrite(p_raw, raw8)
    iio.imwrite(p_pre, pre8)
    out["raw_tif"] = str(p_raw)
    out["preprocessed_tif"] = str(p_pre)

    # binarized / skeletonized は bool か 0/1 を想定して 8bit保存
    bin8 = (img_binarized.astype(np.uint8) * 255)
    sk8 = (img_skeletonized.astype(np.uint8) * 255)

    p_bin = inter_dir / f"{tag}__binarized.tif"
    p_skel = inter_dir / f"{tag}__skeletonized.tif"
    iio.imwrite(p_bin, bin8)
    iio.imwrite(p_skel, sk8)
    out["binarized_tif"] = str(p_bin)
    out["skeletonized_tif"] = str(p_skel)

    return out


def _save_graph_json(g: CompressedGraph, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    nodes = []
    for nid, n in g.nodes.items():
        nodes.append(
            {
                "node_id": int(nid),
                "r": int(n.coord[0]),
                "c": int(n.coord[1]),
                "kind": str(getattr(n, "kind", "")),
                "degree": int(getattr(n, "degree", 0)),
            }
        )

    segs = []
    for sid, s in g.segments.items():
        segs.append(
            {
                "seg_id": int(sid),
                "start_node": int(s.start_node),
                "end_node": int(s.end_node),
                "length_px": float(getattr(s, "length_px", 0.0)),
                "touches_border": bool(getattr(s, "touches_border", False)),
                "n_pixels": int(len(s.pixels)),
            }
        )

    payload = {
        "summary": {
            "nodes": int(len(nodes)),
            "segments": int(len(segs)),
        },
        "nodes": nodes,
        "segments": segs,
        "pair_map_size": int(len(getattr(g, "pair_map", {}) or {})),
    }

    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)


def _find_project_root(any_path: Path) -> Path:
    """
    近い親ディレクトリに data/ がある前提でプロジェクトルートを推定。
    見つからなければカレントを使う。
    """
    p = any_path.resolve()
    for parent in [p] + list(p.parents):
        if (parent / "data").exists():
            return parent
    return Path.cwd().resolve()
