# script/run_inspection.py
from __future__ import annotations

import argparse
import csv
import math
import time
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
    # Bresenham + disk stamp
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
    """
    生成仕様
      - 起点は (0,0) から始まる size*size 領域で乱数
      - 向きはランダム（0..360度）
      - キャンバスは size + (fiber_length*2)
      - はみ出し防止のため、起点をキャンバス上で +fiber_length オフセットして描画
    """
    size = int(size)
    L = int(fiber_length_px)
    canvas = int(size + (L * 2))
    img = np.zeros((canvas, canvas), dtype=np.uint8)

    margin = L  # このマージンにより全方向でも基本はみ出さない

    for _ in range(int(fiber_number)):
        r0_local = int(rng.integers(0, size))
        c0_local = int(rng.integers(0, size))

        r0 = r0_local + margin
        c0 = c0_local + margin

        theta = float(rng.random()) * (2.0 * math.pi)  # 0..2pi
        dr = math.sin(theta)
        dc = math.cos(theta)

        r1 = int(round(r0 + dr * float(L)))
        c1 = int(round(c0 + dc * float(L)))

        # マージン設計上は出ない想定だが、丸めで1pxはみ出す可能性があるので軽くクランプ
        r1 = 0 if r1 < 0 else (canvas - 1 if r1 >= canvas else r1)
        c1 = 0 if c1 < 0 else (canvas - 1 if c1 >= canvas else c1)

        _draw_thick_line(img, (r0, c0), (r1, c1), int(fiber_width_px))

    return img  # 0/1


def length_weighted_stats(lengths_px: np.ndarray) -> Dict[str, float]:
    """
    長さ重みづけ累積% に基づく Rx（%）と、長さ重みづけ平均（ΣL^2/ΣL）、
    total_length（ΣL）を返す。空なら全て0.
    """
    x = np.asarray(lengths_px, dtype=np.float64)
    x = x[np.isfinite(x)]
    x = x[x > 0]

    out: Dict[str, float] = {}
    percentiles = [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 98, 99, 100]

    if x.size == 0:
        for p in percentiles:
            out[f"R{p:02d}"] = 0.0
        out["total_length"] = 0.0
        out["mean_length"] = 0.0
        return out

    order = np.argsort(x)
    xs = x[order]
    w = xs.copy()
    cw = np.cumsum(w)
    total_w = float(cw[-1])

    def R(pct: float) -> float:
        if total_w <= 0.0:
            return 0.0
        t = (pct / 100.0) * total_w
        idx = int(np.searchsorted(cw, t, side="left"))
        if idx < 0:
            idx = 0
        if idx >= xs.size:
            idx = xs.size - 1
        return float(xs[idx])

    for p in percentiles:
        out[f"R{p:02d}"] = R(float(p))

    out["total_length"] = float(np.sum(xs))
    out["mean_length"] = float(np.sum(xs * xs) / np.sum(xs))  # ΣL^2/ΣL
    return out


def run_skeleton_pipeline_and_save(
    *,
    img_bin01: np.ndarray,
    out_dir: Path,
    tag: str,
) -> Dict[str, object]:
    """
    スケルトン以降を実行し、separated_fiber画像を保存する。
    """
    t0 = time.perf_counter()

    img_for_skel = noize_elimination(img_bin01.astype(bool), CFG.eliminate_length_px)
    t_noize = time.perf_counter()

    img_skel = skeletonize(img_for_skel)
    t_skel = time.perf_counter()

    g0 = convert_to_graph(img_skel, CFG.border_margin_px)
    t_g0 = time.perf_counter()

    g1 = merge_nodes(g0, merge_short_seg_px=float(getattr(CFG, "merge_short_seg_px", 3)))
    t_g1 = time.perf_counter()

    seg_before = sum(
        1 for s in g1.segments.values()
        if getattr(s, "kind", SegmentKind.SEGMENT) == SegmentKind.SEGMENT
    )

    g2 = kink_cut(g1, CFG.threshold_of_nonlinear, CFG.blob_px, CFG.cut_max, CFG.cut_angle)
    t_g2 = time.perf_counter()

    seg_after = sum(
        1 for s in g2.segments.values()
        if getattr(s, "kind", SegmentKind.SEGMENT) == SegmentKind.SEGMENT
    )

    g3 = pairing(g2, CFG.pairing_angle_max, CFG.pairing_length_for_calc_angle)
    t_g3 = time.perf_counter()

    fibers = measure_length(g3, CFG.top_cut)
    t_len = time.perf_counter()

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


def _fmt2(x: float) -> str:
    return f"{float(x):.2f}"


def _nan_stats(vals: List[float]) -> Dict[str, float]:
    a = np.array(vals, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"mean": 0.0, "sigma": 0.0, "max": 0.0, "min": 0.0}
    return {
        "mean": float(np.mean(a)),
        "sigma": float(np.std(a, ddof=0)),
        "max": float(np.max(a)),
        "min": float(np.min(a)),
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
    print("steps: generate tif -> skeleton+graph+merge+kink+pair+measure -> separated tif -> write raw csv", flush=True)

    rng = np.random.default_rng(int(args.seed))

    rows: List[Dict[str, str]] = []
    with cond_path.open("r", newline="", encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append({k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in row.items()})

    # 出力CSV（raw）
    out_csv = out_dir / f"{cond_path.stem}_result.csv"

    percent_cols = [f"R{p:02d}" for p in [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 98, 99, 100]]

    header = [
        "test_code",
        "fiber_width",
        "fiber_length",
        "duplication_number",
        "fiber_number",
        "R50",
        "mean_length",
        "fiber_count",
        "total_length",
        "junction_num",
        "separation_num",
        "pairing_num",
    ] + percent_cols

    t_all0 = time.perf_counter()

    with out_csv.open("w", newline="", encoding="utf-8") as fout:
        w = csv.writer(fout)
        w.writerow(header)

        prev_test_code = None

        for row_i, row in enumerate(rows, start=1):
            test_code = str(row["test_code"]).strip()
            fiber_width_px = int(str(row["fiber_width"]).strip())
            fiber_length_px = int(str(row["fiber_length"]).strip())
            fiber_number = int(str(row["fiber_number"]).strip())

            if prev_test_code is not None and test_code != prev_test_code:
                w.writerow([])  # test_codeが変わる部分では1行空ける
            prev_test_code = test_code

            print("", flush=True)
            print(f"[{row_i}/{len(rows)}] test_code={test_code} w={fiber_width_px} l={fiber_length_px} n={fiber_number}", flush=True)

            # 各 duplication の数値を貯める（後で mean/sigma/max/min）
            acc: Dict[str, List[float]] = {k: [] for k in (["R50", "mean_length", "fiber_count", "total_length", "junction_num", "separation_num", "pairing_num"] + percent_cols)}

            for dup in range(1, 6):
                t0 = time.perf_counter()

                img01 = generate_fiber_image(
                    size=int(args.size),
                    fiber_width_px=fiber_width_px,
                    fiber_length_px=fiber_length_px,
                    fiber_number=fiber_number,
                    rng=rng,
                )

                fname = f"test{test_code}_w{fiber_width_px:02d}_l{fiber_length_px:04d}_f{fiber_number:04d}_n{dup}.tif"
                img_path = out_dir / fname

                iio.imwrite(img_path, (img01 * 255).astype(np.uint8))

                tag = img_path.stem
                run = run_skeleton_pipeline_and_save(img_bin01=img01, out_dir=out_dir, tag=tag)

                dt = time.perf_counter() - t0
                timing = run["timing"]
                stats = run["stats"]

                print(
                    f"  dup={dup}/5  {fname}  total={dt:.3f}s "
                    f"(noize={timing['noize']:.3f}s skel={timing['skeletonize']:.3f}s graph={timing['convert_to_graph']:.3f}s "
                    f"merge={timing['merge_nodes']:.3f}s kink={timing['kink_cut']:.3f}s pair={timing['pairing']:.3f}s "
                    f"len={timing['measure_length']:.3f}s draw={timing['draw']:.3f}s)",
                    flush=True,
                )

                row_out: List[str] = [
                    test_code,
                    f"{fiber_width_px:d}",
                    f"{fiber_length_px:d}",
                    f"{dup:d}",
                    f"{fiber_number:d}",
                    _fmt2(float(stats["R50"])),
                    _fmt2(float(stats["mean_length"])),
                    _fmt2(float(run["fiber_count"])),
                    _fmt2(float(stats["total_length"])),
                    _fmt2(float(run["junction_num"])),
                    _fmt2(float(run["separation_num"])),
                    _fmt2(float(run["pairing_num"])),
                ]

                # percentiles
                for k in percent_cols:
                    row_out.append(_fmt2(float(stats[k])))

                w.writerow(row_out)

                # accumulate
                acc["R50"].append(float(stats["R50"]))
                acc["mean_length"].append(float(stats["mean_length"]))
                acc["fiber_count"].append(float(run["fiber_count"]))
                acc["total_length"].append(float(stats["total_length"]))
                acc["junction_num"].append(float(run["junction_num"]))
                acc["separation_num"].append(float(run["separation_num"]))
                acc["pairing_num"].append(float(run["pairing_num"]))
                for k in percent_cols:
                    acc[k].append(float(stats[k]))

            # summary 4行（mean, sigma, max, min）
            for label in ["mean", "sigma", "max", "min"]:
                # 先頭列は埋める（見た目を揃える）
                out_sum = [
                    test_code,
                    f"{fiber_width_px:d}",
                    f"{fiber_length_px:d}",
                    label,  # duplication_number欄に入れる
                    f"{fiber_number:d}",
                ]

                # 指定の主要列 + percentiles を同じ順に出す
                for col in ["R50", "mean_length", "fiber_count", "total_length", "junction_num", "separation_num", "pairing_num"] + percent_cols:
                    st = _nan_stats(acc[col])
                    out_sum.append(_fmt2(float(st[label])))

                w.writerow(out_sum)

    dt_all = time.perf_counter() - t_all0
    print("", flush=True)
    print(f"[DONE] wrote: {out_csv}  total_time={dt_all:.3f}s", flush=True)


if __name__ == "__main__":
    main()
