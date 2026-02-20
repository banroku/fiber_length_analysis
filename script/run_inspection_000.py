# script/run_inspection.py
from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import imageio.v3 as iio

from fiberlen.config import CFG
from fiberlen.noize_elimination import noize_elimination
from fiberlen.skeletonize import skeletonize
from fiberlen.convert_to_graph import convert_to_graph
from fiberlen.merge_nodes import merge_nodes
from fiberlen.kink_cut import kink_cut
from fiberlen.pairing import pairing
from fiberlen.measure_length import measure_length
from fiberlen.draw_separated_fiber_img import configure_draw_output, draw_separated_fiber_img
from fiberlen.types import NodeKind, SegmentKind


Pixel = Tuple[int, int]


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _disk_offsets(radius: int) -> List[Pixel]:
    r = int(radius)
    if r <= 0:
        return [(0, 0)]
    out: List[Pixel] = []
    rr2 = r * r
    for dr in range(-r, r + 1):
        for dc in range(-r, r + 1):
            if dr * dr + dc * dc <= rr2:
                out.append((dr, dc))
    return out


def _draw_thick_line(img: np.ndarray, p0: Pixel, p1: Pixel, width_px: int) -> None:
    h, w = img.shape
    x0, y0 = int(p0[1]), int(p0[0])
    x1, y1 = int(p1[1]), int(p1[0])

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    rad = max(0, int(round(width_px / 2.0)))
    stamp = _disk_offsets(rad)

    while True:
        for dr, dc in stamp:
            rr = y0 + dr
            cc = x0 + dc
            if 0 <= rr < h and 0 <= cc < w:
                img[rr, cc] = 1

        if x0 == x1 and y0 == y1:
            break

        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy


def generate_fiber_image(
    *,
    size: int,
    fiber_width_px: int,
    fiber_length_px: int,
    fiber_number: int,
    rng: np.random.Generator,
) -> np.ndarray:
    # キャンバスは size + fiber_length（はみ出し防止）
    canvas = int(size + fiber_length_px)
    img = np.zeros((canvas, canvas), dtype=np.uint8)

    # 起点は (0,0) から始まる size*size 領域にランダム配置
    # 向きは x,y とも正方向（角度 0..90度、dx>=0, dy>=0）
    for _ in range(int(fiber_number)):
        r0 = int(rng.integers(0, size))
        c0 = int(rng.integers(0, size))

        theta = float(rng.random()) * (math.pi / 2.0)  # 0..90deg
        dr = math.sin(theta)
        dc = math.cos(theta)

        r1 = int(round(r0 + dr * float(fiber_length_px)))
        c1 = int(round(c0 + dc * float(fiber_length_px)))

        # 念のため canvas に収める（正方向限定なので基本不要だが丸め誤差対策）
        r1 = min(max(r1, 0), canvas - 1)
        c1 = min(max(c1, 0), canvas - 1)

        _draw_thick_line(img, (r0, c0), (r1, c1), int(fiber_width_px))

    # 0/1 のバイナリ画像（tif保存は 0/255 にする）
    return img


def length_weighted_stats(lengths_px: np.ndarray) -> Dict[str, float]:
    # lengths_px: shape (n,), >=0
    if lengths_px.size == 0:
        out = {"mean_length": 0.0, "total_length": 0.0}
        for p in [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]:
            out[f"R{p}"] = 0.0
        return out

    x = np.asarray(lengths_px, dtype=np.float64)
    x = x[x > 0]
    if x.size == 0:
        out = {"mean_length": 0.0, "total_length": 0.0}
        for p in [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]:
            out[f"R{p}"] = 0.0
        return out

    order = np.argsort(x)
    xs = x[order]
    w = xs.copy()  # 長さ重み（=長さ）
    cw = np.cumsum(w)
    total_w = float(cw[-1])

    def R(pct: float) -> float:
        # 「長さ重みづけ累積%」が pct に到達する長さ
        if total_w <= 0:
            return 0.0
        t = (pct / 100.0) * total_w
        idx = int(np.searchsorted(cw, t, side="left"))
        if idx < 0:
            idx = 0
        if idx >= xs.size:
            idx = xs.size - 1
        return float(xs[idx])

    out: Dict[str, float] = {}
    for p in [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]:
        out[f"R{p}"] = R(float(p))

    out["total_length"] = float(np.sum(xs))
    out["mean_length"] = float(np.sum(xs * xs) / np.sum(xs))  # 長さ重みづけ平均 = ΣL^2 / ΣL
    return out


def run_skeleton_pipeline_and_save(
    *,
    img_bin01: np.ndarray,
    out_dir: Path,
    tag: str,
) -> Dict[str, object]:
    # bin -> noize -> skeleton -> graph -> merge -> kink -> pairing -> measure -> draw
    t0 = time.perf_counter()
    img_for_skel = noize_elimination(img_bin01.astype(bool), CFG.eliminate_length_px)
    t_noize = time.perf_counter()

    img_skel = skeletonize(img_for_skel)
    t_skel = time.perf_counter()

    g0 = convert_to_graph(img_skel, CFG.border_margin_px)
    t_g0 = time.perf_counter()

    # GUI反復前提なら merge_nodes は非破壊版が望ましい（入力g0を汚さない実装前提）
    g1 = merge_nodes(g0, merge_short_seg_px=float(getattr(CFG, "merge_short_seg_px", 3)))
    t_g1 = time.perf_counter()

    seg_before = sum(1 for s in g1.segments.values() if getattr(s, "kind", SegmentKind.SEGMENT) == SegmentKind.SEGMENT)

    g2 = kink_cut(g1, CFG.threshold_of_nonlinear, CFG.blob_px, CFG.cut_max, CFG.cut_angle)
    t_g2 = time.perf_counter()

    seg_after = sum(1 for s in g2.segments.values() if getattr(s, "kind", SegmentKind.SEGMENT) == SegmentKind.SEGMENT)

    g3 = pairing(g2, CFG.pairing_angle_max, CFG.pairing_length_for_calc_angle)
    t_g3 = time.perf_counter()

    fibers = measure_length(g3, CFG.top_cut)
    t_len = time.perf_counter()

    # draw_separated_fiber_img は out_dir/tag をグローバル設定して保存する方式
    configure_draw_output(out_dir, tag)
    draw_separated_fiber_img(g3, img_skel)
    t_draw = time.perf_counter()

    pair_map = getattr(g3, "pair_map", {}) or {}
    pairing_num = int(len(pair_map))
    junction_num = int(sum(1 for n in g3.nodes.values() if getattr(n, "kind", None) == NodeKind.JUNCTION))
    separation_num = int(seg_after - seg_before)

    lengths_px = np.array([float(getattr(f, "length_px", 0.0)) for f in fibers], dtype=np.float64)
    stats = length_weighted_stats(lengths_px)

    return {
        "img_skeletonized": img_skel,
        "graph": g3,
        "fibers": fibers,
        "fiber_count": int(len(fibers)),
        "junction_num": junction_num,
        "separation_num": separation_num,
        "pairing_num": pairing_num,
        "stats": stats,
        "timing": {
            "noize": float(t_noize - t0),
            "skeletonize": float(t_skel - t_noize),
            "convert_to_graph": float(t_g0 - t_skel),
            "merge_nodes": float(t_g1 - t_g0),
            "kink_cut": float(t_g2 - t_g1),
            "pairing": float(t_g3 - t_g2),
            "measure_length": float(t_len - t_g3),
            "draw": float(t_draw - t_len),
            "total": float(t_draw - t0),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("conditions_csv", type=str)
    ap.add_argument("--size", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cond_path = Path(args.conditions_csv)
    out_dir = Path("data") / "inspection"
    _ensure_dir(out_dir)

    print("=== Inspection runner ===", flush=True)
    print(f"conditions_csv={cond_path}", flush=True)
    print(f"output_dir={out_dir}", flush=True)
    print(f"size={args.size}  n_per_condition=5  seed={args.seed}", flush=True)
    print("steps: generate tif -> skeleton+graph+merge+kink+pair+measure -> separated tif -> aggregate csv", flush=True)

    rng = np.random.default_rng(int(args.seed))

    rows: List[Dict[str, str]] = []
    with cond_path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)

    results: List[Dict[str, object]] = []

    t_all0 = time.perf_counter()
    for row_i, row in enumerate(rows, start=1):
        test_code = str(row["test_code"]).strip()
        fiber_width_px = int(str(row["fiber_width"]).strip())
        fiber_length_px = int(str(row["fiber_length"]).strip())
        fiber_number = int(str(row["fiber_number"]).strip())

        print("", flush=True)
        print(f"[{row_i}/{len(rows)}] test_code={test_code} w={fiber_width_px} l={fiber_length_px} n={fiber_number}", flush=True)

        for b in range(1, 6):  # n=5
            t0 = time.perf_counter()

            img01 = generate_fiber_image(
                size=int(args.size),
                fiber_width_px=fiber_width_px,
                fiber_length_px=fiber_length_px,
                fiber_number=fiber_number,
                rng=rng,
            )

            # 保存名
            fname = f"test{test_code}_w{fiber_width_px:02d}_l{fiber_length_px:04d}_f{fiber_number:04d}_n{b}.tif"
            img_path = out_dir / fname

            # tif保存（0/255）
            iio.imwrite(img_path, (img01 * 255).astype(np.uint8))

            tag = img_path.stem

            # スケルトン以降の処理
            run = run_skeleton_pipeline_and_save(img_bin01=img01, out_dir=out_dir, tag=tag)

            dt = time.perf_counter() - t0
            timing = run["timing"]
            stats = run["stats"]

            print(
                f"  n={b}/5  {fname}  total={dt:.3f}s "
                f"(noize={timing['noize']:.3f}s skel={timing['skeletonize']:.3f}s graph={timing['convert_to_graph']:.3f}s "
                f"merge={timing['merge_nodes']:.3f}s kink={timing['kink_cut']:.3f}s pair={timing['pairing']:.3f}s "
                f"len={timing['measure_length']:.3f}s draw={timing['draw']:.3f}s)",
                flush=True,
            )

            results.append(
                {
                    "test_code": test_code,
                    "fiber_width": fiber_width_px,
                    "fiber_length": fiber_length_px,
                    "fiber_number": fiber_number,
                    "replicate_n": b,
                    "generated_tif": str(img_path),
                    "separated_tif": str(out_dir / f"{tag}__paired_segments.tif"),
                    "fiber_count": int(run["fiber_count"]),
                    "total_length": float(stats["total_length"]),
                    "mean_length": float(stats["mean_length"]),
                    "junction_num": int(run["junction_num"]),
                    "separation_num": int(run["separation_num"]),
                    "pairing_num": int(run["pairing_num"]),
                    "R5": float(stats["R5"]),
                    "R10": float(stats["R10"]),
                    "R20": float(stats["R20"]),
                    "R30": float(stats["R30"]),
                    "R40": float(stats["R40"]),
                    "R50": float(stats["R50"]),
                    "R60": float(stats["R60"]),
                    "R70": float(stats["R70"]),
                    "R80": float(stats["R80"]),
                    "R90": float(stats["R90"]),
                    "R95": float(stats["R95"]),
                    "R100": float(stats["R100"]),
                }
            )

    # 集計CSV出力
    out_csv = out_dir / f"{cond_path.stem}_result.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "test_code",
                "fiber_width",
                "fiber_length",
                "fiber_number",
                "replicate_n",
                "generated_tif",
                "separated_tif",
                "R5",
                "R10",
                "R20",
                "R30",
                "R40",
                "R50",
                "R60",
                "R70",
                "R80",
                "R90",
                "R95",
                "R100",
                "mean_length",
                "fiber_count",
                "total_length",
                "junction_num",
                "separation_num",
                "pairing_num",
            ],
        )
        w.writeheader()
        for r in results:
            w.writerow(r)

    dt_all = time.perf_counter() - t_all0
    print("", flush=True)
    print(f"[DONE] wrote: {out_csv}  total_time={dt_all:.3f}s", flush=True)


if __name__ == "__main__":
    main()
