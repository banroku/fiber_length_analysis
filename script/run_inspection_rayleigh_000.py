# script/run_inspection_rayleigh.py
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


def _rayleigh_cdf(x: np.ndarray, sigma: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    sig2 = float(sigma) * float(sigma)
    y = 1.0 - np.exp(-(x * x) / (2.0 * sig2))
    y[x < 0] = 0.0
    return y


def _rayleigh_ppf(p: np.ndarray, sigma: float) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, 0.0, 1.0 - 1e-15)
    return float(sigma) * np.sqrt(-2.0 * np.log1p(-p))


def _rayleigh_cond_mean(a: float, b: float, sigma: float) -> float:
    """
    条件付き平均 E[X | a<=X<b] を数値積分で近似（軽量・安定）。
    a,b は px スケール。b は有限を想定（最後のビンも有限にする設計）。
    """
    a = float(a)
    b = float(b)
    if b <= a:
        return max(1.0, a)

    # 64分割の台形で十分
    xs = np.linspace(a, b, 64, dtype=np.float64)
    # pdf: (x/sigma^2)*exp(-x^2/(2sigma^2))
    sig2 = float(sigma) * float(sigma)
    pdf = (xs / sig2) * np.exp(-(xs * xs) / (2.0 * sig2))
    num = np.trapezoid(xs * pdf, xs)
    den = np.trapezoid(pdf, xs)
    if den <= 0:
        return 0.5 * (a + b)
    return float(num / den)


def _rayleigh_bins_20(mean_target: float, base_sigma: float = 2.0) -> Dict[str, object]:
    """
    20区間を作る。
    基底 Rayleigh(sigma=base_sigma) を平均が mean_target になるようにスケールする。

    ビンは 0?p99.5 までを等間隔19分割し、最後の1ビンも同幅で p99.5?p99.5+width とする（有限）。
    それぞれの区間に対して
      - 理論出現比率（確率質量）
      - 区間の条件付き平均長
    を返す。
    """
    mean_target = float(mean_target)
    base_sigma = float(base_sigma)

    base_mean = base_sigma * math.sqrt(math.pi / 2.0)
    scale = mean_target / base_mean
    sigma_eff = base_sigma * scale  # スケール後は Rayleigh(sigma_eff) と同等
    # 理論平均は mean_target（数値誤差無視）

    p_hi = 0.995
    x_hi = float(_rayleigh_ppf(np.array([p_hi]), sigma_eff)[0])
    width = x_hi / 19.0  # 0..x_hi を19区間 → 19本の境界幅

    edges = np.array([i * width for i in range(20)] + [x_hi], dtype=np.float64)
    # edges は 0, w, 2w, ... 19w (=x_hi), x_hi（最後が重複）になりやすいので整える
    edges = np.array([i * width for i in range(20)] + [x_hi + width], dtype=np.float64)  # 21個: 0..20
    # 20区間: [0..w), ... , [19w..20w) で最後も有限にする

    cdf_edges = _rayleigh_cdf(edges, sigma_eff)
    probs = np.diff(cdf_edges)  # 20個
    probs = np.maximum(probs, 0.0)
    s = float(np.sum(probs))
    if s > 0:
        probs = probs / s  # 有限区間内での相対比率（尾部は最後ビンに吸収されないが、比較用に正規化）

    means = np.zeros(20, dtype=np.float64)
    for i in range(20):
        means[i] = _rayleigh_cond_mean(edges[i], edges[i + 1], sigma_eff)
    means = np.maximum(means, 1.0)

    return {
        "sigma_eff": sigma_eff,
        "edges": edges,        # 21
        "probs": probs,        # 20 (正規化後)
        "bin_means": means,    # 20
        "theory_mean": mean_target,
    }


def generate_fiber_image_rayleigh_binned(
    *,
    size: int,
    fiber_width_px: int,
    fiber_length_mean_px: int,
    fiber_number: int,
    rng: np.random.Generator,
    n_bins: int = 20,
) -> Dict[str, object]:
    """
    生成仕様（Rayleigh 20区間近似）
      - fiber_length_mean_px を理論平均とする Rayleigh を採用（基底sigma=2をスケール）
      - 20区間に分割し、各区間の条件付き平均長で繊維長を代表
      - 総延長 = fiber_length_mean_px * fiber_number を固定し、その総延長を各ビンへ理論比率で配分
      - 配分された延長を各ビン平均長で割って本数を決める（整数化）
      - 起点は (0,0) から始まる size*size 領域で乱数
      - 向きはランダム（0..360度）
      - キャンバスは size + (2*fiber_length_mean_px)
    """
    size = int(size)
    mean_L = int(fiber_length_mean_px)
    total_len_target = float(mean_L * int(fiber_number))

    canvas = int(size + (mean_L * 2))
    img = np.zeros((canvas, canvas), dtype=np.uint8)
    margin = mean_L

    bins = _rayleigh_bins_20(mean_target=float(mean_L), base_sigma=2.0)
    probs = np.asarray(bins["probs"], dtype=np.float64)  # 20
    bin_means = np.asarray(bins["bin_means"], dtype=np.float64)

    # 各ビンに割り当てる総延長（理論比率）
    alloc_len = total_len_target * probs

    # 各ビン本数（期待総延長を満たすように）
    counts = np.floor(alloc_len / bin_means + 0.5).astype(int)
    counts = np.maximum(counts, 0)

    # 端数調整（総延長が目標に近づくように、誤差に応じて最大比率ビンへ加減）
    def total_len_from_counts(cc: np.ndarray) -> float:
        return float(np.sum(cc.astype(np.float64) * bin_means))

    cur_total = total_len_from_counts(counts)
    # 誤差が大きい場合だけ、最大probビンで調整
    idx_main = int(np.argmax(probs))
    if bin_means[idx_main] > 0:
        diff = total_len_target - cur_total
        step = int(round(diff / float(bin_means[idx_main])))
        if step != 0:
            counts[idx_main] = max(0, int(counts[idx_main] + step))

    lengths_to_draw: List[int] = []
    for i in range(n_bins):
        Lb = int(round(float(bin_means[i])))
        if Lb < 1:
            Lb = 1
        lengths_to_draw.extend([Lb] * int(counts[i]))

    # 実描画（順序をシャッフルすると見た目が偏りにくい）
    rng.shuffle(lengths_to_draw)

    for L in lengths_to_draw:
        r0_local = int(rng.integers(0, size))
        c0_local = int(rng.integers(0, size))
        r0 = r0_local + margin
        c0 = c0_local + margin

        theta = float(rng.random()) * (2.0 * math.pi)
        dr = math.sin(theta)
        dc = math.cos(theta)

        r1 = int(round(r0 + dr * float(L)))
        c1 = int(round(c0 + dc * float(L)))

        # 軽いクランプ
        r1 = 0 if r1 < 0 else (canvas - 1 if r1 >= canvas else r1)
        c1 = 0 if c1 < 0 else (canvas - 1 if c1 >= canvas else c1)

        _draw_thick_line(img, (r0, c0), (r1, c1), int(fiber_width_px))

    return {
        "img01": img,
        "bins": bins,
        "generated_lengths_px": np.array(lengths_to_draw, dtype=np.float64),
        "total_len_target": float(total_len_target),
        "generated_fiber_count": int(len(lengths_to_draw)),
    }


def run_skeleton_pipeline_and_save(*, img_bin01: np.ndarray, out_dir: Path, tag: str) -> Dict[str, object]:
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
    lengths_px = lengths_px[np.isfinite(lengths_px)]
    lengths_px = lengths_px[lengths_px > 0]

    if lengths_px.size == 0:
        measured_mean = 0.0
    else:
        measured_mean = float(np.mean(lengths_px))

    return {
        "fibers": fibers,
        "fiber_count": int(len(fibers)),
        "junction_num": junction_num,
        "separation_num": separation_num,
        "pairing_num": pairing_num,
        "measured_mean_length": measured_mean,
        "measured_lengths_px": lengths_px,
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


def _bin_ratios(lengths_px: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """
    edges: 21個。20区間。
    戻り: 20個（個数比率）
    """
    x = np.asarray(lengths_px, dtype=np.float64)
    x = x[np.isfinite(x)]
    x = x[x >= 0]
    if x.size == 0:
        return np.zeros(20, dtype=np.float64)

    counts, _ = np.histogram(x, bins=edges)
    s = float(np.sum(counts))
    if s <= 0:
        return np.zeros(20, dtype=np.float64)
    return counts.astype(np.float64) / s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("conditions_csv", type=str)
    ap.add_argument("--size", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cond_path = Path(args.conditions_csv)
    out_dir = Path("data") / "inspection"
    _ensure_dir(out_dir)

    print("=== Inspection runner (Rayleigh) ===", flush=True)
    print(f"conditions_csv={cond_path}", flush=True)
    print(f"output_dir={out_dir}", flush=True)
    print(f"size={args.size}  n_per_condition=5  seed={args.seed}", flush=True)
    print("steps: generate(rayleigh bins) tif -> skeleton+graph+merge+kink+pair+measure -> separated tif -> write raw csv", flush=True)

    rng = np.random.default_rng(int(args.seed))

    rows: List[Dict[str, str]] = []
    with cond_path.open("r", newline="", encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append({k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in row.items()})

    out_csv = out_dir / f"{cond_path.stem}_result_rayleigh_raw.csv"

    bin_cols = [f"bin{idx:02d}_ratio" for idx in range(20)]
    header = [
        "test_code",
        "fiber_width",
        "fiber_length_mean",
        "duplication_number",
        "fiber_number",
        "mean_length_measured",
    ] + bin_cols

    t_all0 = time.perf_counter()

    with out_csv.open("w", newline="", encoding="utf-8") as fout:
        w = csv.writer(fout)
        w.writerow(header)

        prev_test_code = None

        for row_i, row in enumerate(rows, start=1):
            test_code = str(row["test_code"]).strip()
            fiber_width_px = int(str(row["fiber_width"]).strip())
            fiber_mean_px = int(str(row["fiber_length"]).strip())
            fiber_number = int(str(row["fiber_number"]).strip())

            if prev_test_code is not None and test_code != prev_test_code:
                w.writerow([])
            prev_test_code = test_code

            print("", flush=True)
            print(f"[{row_i}/{len(rows)}] test_code={test_code} w={fiber_width_px} meanL={fiber_mean_px} n={fiber_number}", flush=True)

            # 理論（この test_code 固有）
            bins = _rayleigh_bins_20(mean_target=float(fiber_mean_px), base_sigma=2.0)
            edges = bins["edges"]
            theory_probs = np.asarray(bins["probs"], dtype=np.float64)  # 20（正規化）
            theory_mean = float(bins["theory_mean"])  # = fiber_mean_px

            # duplicationごとの値を貯める（mean と bin比率）
            acc: Dict[str, List[float]] = {"mean_length_measured": []}
            for k in bin_cols:
                acc[k] = []

            for dup in range(1, 6):
                t0 = time.perf_counter()

                gen = generate_fiber_image_rayleigh_binned(
                    size=int(args.size),
                    fiber_width_px=fiber_width_px,
                    fiber_length_mean_px=fiber_mean_px,
                    fiber_number=fiber_number,
                    rng=rng,
                    n_bins=20,
                )

                img01 = gen["img01"]

                fname = f"test{test_code}_w{fiber_width_px:02d}_l{fiber_mean_px:04d}_f{fiber_number:04d}_n{dup}.tif"
                img_path = out_dir / fname
                iio.imwrite(img_path, (img01 * 255).astype(np.uint8))

                tag = img_path.stem
                run = run_skeleton_pipeline_and_save(img_bin01=img01, out_dir=out_dir, tag=tag)

                measured_mean = float(run["measured_mean_length"])
                ratios = _bin_ratios(run["measured_lengths_px"], edges=edges)

                dt = time.perf_counter() - t0
                timing = run["timing"]
                print(
                    f"  dup={dup}/5  {fname}  total={dt:.3f}s "
                    f"(noize={timing['noize']:.3f}s skel={timing['skeletonize']:.3f}s graph={timing['convert_to_graph']:.3f}s "
                    f"merge={timing['merge_nodes']:.3f}s kink={timing['kink_cut']:.3f}s pair={timing['pairing']:.3f}s "
                    f"len={timing['measure_length']:.3f}s draw={timing['draw']:.3f}s)",
                    flush=True,
                )

                row_out = [
                    test_code,
                    f"{fiber_width_px:d}",
                    f"{fiber_mean_px:d}",
                    f"{dup:d}",
                    f"{fiber_number:d}",
                    _fmt2(measured_mean),
                ] + [_fmt2(float(r)) for r in ratios]

                w.writerow(row_out)

                acc["mean_length_measured"].append(measured_mean)
                for i in range(20):
                    acc[bin_cols[i]].append(float(ratios[i]))

            # summary（mean, sigma, max, min）
            for label in ["mean", "sigma", "max", "min"]:
                out_sum = [
                    test_code,
                    f"{fiber_width_px:d}",
                    f"{fiber_mean_px:d}",
                    label,
                    f"{fiber_number:d}",
                ]
                st = _nan_stats(acc["mean_length_measured"])
                out_sum.append(_fmt2(float(st[label])))

                for i in range(20):
                    st_i = _nan_stats(acc[bin_cols[i]])
                    out_sum.append(_fmt2(float(st_i[label])))

                w.writerow(out_sum)

            # theory line（理論値）
            out_theory = [
                test_code,
                f"{fiber_width_px:d}",
                f"{fiber_mean_px:d}",
                "theory",
                f"{fiber_number:d}",
                _fmt2(theory_mean),
            ] + [_fmt2(float(p)) for p in theory_probs]
            w.writerow(out_theory)

    dt_all = time.perf_counter() - t_all0
    print("", flush=True)
    print(f"[DONE] wrote: {out_csv}  total_time={dt_all:.3f}s", flush=True)


if __name__ == "__main__":
    main()
