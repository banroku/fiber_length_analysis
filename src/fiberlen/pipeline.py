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
from typing import Any, Dict, Optional
from collections import Counter

import json
import time

import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt

from fiberlen.config import CFG

from fiberlen.input_img import input_img
from fiberlen.subtract_background import subtract_background
from fiberlen.calc_otsu_threshold import calc_Otsu_threshold
from fiberlen.binarize import binarize
from fiberlen.noize_elimination import noize_elimination
from fiberlen.skeletonize import skeletonize
from fiberlen.merge_nodes import merge_nodes 
from fiberlen.convert_to_graph import convert_to_graph
from fiberlen.kink_cut import kink_cut
from fiberlen.pairing import pairing
from fiberlen.measure_length import measure_length
from fiberlen.postprocess import postprocess
from fiberlen.types import CompressedGraph, NodeKind, SegmentKind
from fiberlen.visualize_results import configure_visualize_output, visualize_results
from fiberlen.draw_separated_fiber_img import draw_separated_fiber_img

# 結果表示用
def summarize_graph_kinds(graph): 
    node_counter = Counter(n.kind for n in graph.nodes.values())
    seg_counter = Counter(s.kind for s in graph.segments.values())

    node_part = ", ".join(
        f"{k.name}:{node_counter.get(k,0)}" for k in NodeKind
    )
    seg_part = ", ".join(
        f"{k.name}:{seg_counter.get(k,0)}" for k in SegmentKind
    )

    return (
        f"nodes={len(graph.nodes)} [{node_part}] "
        f"segs={len(graph.segments)} [{seg_part}]"
    )

def run_pipeline(
    img_path: str | Path,
    *,
    save_intermediate: bool = True,
    out_tag: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    指定画像1枚を、あなたの設計順で一気通貫に処理する。
    Returns には、最終fiberと、推奨Otsu閾値、出力ファイルパスなどを入れる。
    """
    t_all0 = time.perf_counter()

    def log(msg: str) -> None:
        if verbose:
            print(msg, flush=True)

    def step_begin(name: str) -> float:
        log(f"[BEGIN] {name}")
        return time.perf_counter()

    def step_end(name: str, t0: float, extra: str = "") -> None:
        dt = time.perf_counter() - t0
        if extra:
            log(f"[END]   {name}  {dt:.3f}s  {extra}")
        else:
            log(f"[END]   {name}  {dt:.3f}s")

    img_path = Path(img_path)

    t0 = step_begin("0) resolve paths")
    project_root = _find_project_root(img_path)
    data_dir = project_root / "data"
    input_dir = data_dir / "input"
    inter_dir = data_dir / "intermediate"
    out_dir = data_dir / "output"

    inter_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = img_path.stem
    tag = out_tag.strip() if isinstance(out_tag, str) and out_tag.strip() else stem
    step_end("0) resolve paths", t0, extra=f"tag={tag} root={project_root}")

    # ---- 1) 画像の読み込み ----
    t0 = step_begin("1) input_img")
    img_raw01 = input_img(img_path, CFG.background_is_dark)
    step_end("1) input_img", t0, extra=f"shape={tuple(img_raw01.shape)} dtype={img_raw01.dtype}")

    # ---- 2) 背景均一化 ----
    t0 = step_begin("2) subtract_background")
    img_preprocessed01 = subtract_background(img_raw01, CFG.blur_sigma_px)
    step_end("2) subtract_background", t0, extra=f"shape={tuple(img_preprocessed01.shape)}")
    iio.imwrite(Path("data/intermediate") / f"{Path(img_path).stem}_img_preprocessed.tif", img_preprocessed01) 

    # ---- 3) Otsu推奨閾値 ----
    t0 = step_begin("3) calc_Otsu_threshold")
    threshold_otsu = float(calc_Otsu_threshold(img_preprocessed01))
    step_end("3) calc_Otsu_threshold", t0, extra=f"otsu={threshold_otsu:.6f}")

    # ---- 4) 二値化 ----
    t0 = step_begin("4) binarize")
    img_binarized = binarize(img_preprocessed01, threshold_otsu-0.02)
    # True/False or 0/1 でも状況が分かるように出す
    nz = int(np.count_nonzero(img_binarized))
    step_end("4) binarize", t0, extra=f"nonzero={nz} ({nz/img_binarized.size:.6f}) dtype={img_binarized.dtype}")
    iio.imwrite(Path("data/intermediate") / f"{Path(img_path).stem}_img_binarized.tif", img_binarized) 

    # ---- 5) ノイズ除去 ----
    t0 = step_begin("5) noize_elimination")
    img_for_skeletonize = noize_elimination(img_binarized, CFG.eliminate_length_px) 
    nz2 = int(np.count_nonzero(img_for_skeletonize))
    step_end("5) noize_elimination", t0, extra=f"nonzero={nz2} ({nz2/img_for_skeletonize.size:.6f})")
    iio.imwrite(Path("data/intermediate") / f"{Path(img_path).stem}_img_for_skeletonize.tif", img_for_skeletonize) 

    # ---- 6) スケルトナイズ ----
    t0 = step_begin("6) skeletonize")
    img_skeletonized = skeletonize(img_for_skeletonize)
    nz3 = int(np.count_nonzero(img_skeletonized))
    step_end("6) skeletonize", t0, extra=f"nonzero={nz3} ({nz3/img_skeletonized.size:.6f})")

    # ---- 7) グラフ化 ----
    t0 = step_begin("7) convert_to_graph")
    graph = convert_to_graph(img_skeletonized, CFG.border_margin_px)
    # summaryは graph 側にある前提ではなく、最低限の数だけ出す
    step_end(
        "7) convert_to_graph",
        t0,
        extra=summarize_graph_kinds(graph), 
    )
    # ---- 8.5) 近接ノードのマージ ----
    t0 = step_begin("8.5) merge_nodes")
    graph_nodes_merged = merge_nodes(graph, merge_short_seg_px)
    step_end(
        "8) merge_nodes",
        t0,
        extra=summarize_graph_kinds(graph_nodes_merged), 
    )

    # ---- 8) キンク処理 ----
    t0 = step_begin("8) kink_cut")
    graph_kink_cut = kink_cut(
        graph_nodes_merged,
        CFG.threshold_of_nonlinear,
        CFG.blob_px,
        CFG.cut_max,
        CFG.cut_angle,
    )
    step_end(
        "8) kink_cut",
        t0,
        extra=summarize_graph_kinds(graph_kink_cut), 
    )

    # ---- 9) pairing ----
    t0 = step_begin("9) pairing")
    graph_paired = pairing(
        graph_kink_cut,
        CFG.pairing_angle_max,
        CFG.pairing_length_for_calc_angle,
    )
    pm = getattr(graph_paired, "pair_map", {}) or {}
    step_end(
        "9) pairing",
        t0,
        extra=summarize_graph_kinds(graph_paired), 
    )

    # ---- 10) 長さ測定 ----
    t0 = step_begin("10) measure_length")
    fibers = measure_length(graph_paired, CFG.top_cut)
    step_end("10) measure_length", t0, extra=f"n_fibers={len(fibers)}")

    # ---- 11) postprocess（csv保存）----
    t0 = step_begin("11) postprocess")
    fibers_csv = out_dir / f"{tag}__fibers.csv"
    used_cfg_json = out_dir / f"{tag}__used_config.json"
    fibers_filtered = postprocess(
        fibers,
        CFG.post_eliminate_length_px,
        out_csv_path=fibers_csv,
        out_config_json_path=used_cfg_json,
        used_config=_cfg_payload(threshold_otsu=threshold_otsu),
    )
    step_end("11) postprocess", t0, extra=f"n_final={len(fibers_filtered)} csv={fibers_csv.name}")

    # ---- 11.5) visualize_results（長さ重みづけ統計・ヒストグラム）----
    t0 = step_begin("11.5) visualize_results")
    configure_visualize_output(out_dir, tag)
    #visualize_results(fibers_filtered, hist_range=(0.0, 1500.0), hist_bins=30)
    visualize_results(fibers_filtered, hist_range=CFG.hist_range, hist_bins=CFG.hist_bins)
    step_end("11.5) visualize_results", t0)

#    # ---- 12) show results (saved files only) ----
#
#    print("\n=== Length-weighted summary ===")
#    print((out_dir / f"{tag}__summary.txt").read_text(encoding="utf-8"))
#
#    print("\n=== Length-weighted R table ===")
#    print((out_dir / f"{tag}__R_table_length_weighted.csv").read_text(encoding="utf-8"))
#
#    img = iio.imread(out_dir / f"{tag}__hist_length_weighted.png")
#    plt.imshow(img)
#    plt.axis("off")
#    plt.title("Length-weighted histogram")
#    plt.show()

    # ---- 13) save image ----
    draw_separated_fiber_img(graph_paired, img_skeletonized)





    # ---- optional: intermediate 保存 ----
    saved = {}
    if save_intermediate:
        t0 = step_begin("12) save_intermediate")
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
        gpath = inter_dir / f"{tag}__graph.json"
        _save_graph_json(graph_paired, gpath)
        saved["graph_json"] = str(gpath)
        step_end("12) save_intermediate", t0, extra=f"n_files={len(saved)}")

    dt_all = time.perf_counter() - t_all0
    log(f"[DONE] pipeline total {dt_all:.3f}s")

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

    raw8 = np.clip(img_raw01 * 255.0, 0, 255).astype(np.uint8)
    pre8 = np.clip(img_preprocessed01 * 255.0, 0, 255).astype(np.uint8)

    p_raw = inter_dir / f"{tag}__raw.tif"
    p_pre = inter_dir / f"{tag}__preprocessed.tif"
    iio.imwrite(p_raw, raw8)
    iio.imwrite(p_pre, pre8)
    out["raw_tif"] = str(p_raw)
    out["preprocessed_tif"] = str(p_pre)

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
    p = any_path.resolve()
    for parent in [p] + list(p.parents):
        if (parent / "data").exists():
            return parent
    return Path.cwd().resolve()
