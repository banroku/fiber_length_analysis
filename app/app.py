# app/app.py
import sys
from pathlib import Path
import tempfile
import hashlib

import numpy as np
import streamlit as st
import imageio.v3 as iio
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import fiberlen.config as cfg_mod
from fiberlen.config import CFG, Config

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

from fiberlen.draw_separated_fiber_img import configure_draw_output, draw_separated_fiber_img

st.set_page_config(layout="wide")
st.title("Fiber Length Analysis GUI")


def _file_id_from_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()[:12]


def crop_center(img: np.ndarray, size: int = 800) -> np.ndarray:
    if img.ndim < 2:
        return img
    h, w = img.shape[0], img.shape[1]
    if h <= size and w <= size:
        return img
    r0 = max(0, (h - size) // 2)
    c0 = max(0, (w - size) // 2)
    r1 = min(h, r0 + size)
    c1 = min(w, c0 + size)
    return img[r0:r1, c0:c1, ...]


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    if v.size == 0:
        return float("nan")
    idx = np.argsort(v)
    v = v[idx]
    w = w[idx]
    cum = np.cumsum(w)
    total = float(cum[-1])
    if total <= 0:
        return float("nan")
    target = q * total
    pos = int(np.searchsorted(cum, target, side="left"))
    pos = max(0, min(pos, v.size - 1))
    return float(v[pos])


def fibers_to_lengths_um(fibers, um_per_px: float) -> np.ndarray:
    return np.array([float(f.length_px) * float(um_per_px) for f in fibers], dtype=float)


def compute_upper(img_path: str, background_is_dark: bool, blur_sigma_px: float):
    img_raw01 = input_img(img_path, background_is_dark)
    img_pre01 = subtract_background(img_raw01, blur_sigma_px)
    thr_otsu = float(calc_Otsu_threshold(img_pre01))
    return img_pre01, thr_otsu


def compute_binarized(img_pre01: np.ndarray, threshold: float):
    return binarize(img_pre01, float(threshold))


def compute_lower_and_save(
    img_pre01: np.ndarray,
    threshold_manual: float,
    threshold_otsu: float,
    *,
    tag: str,
    um_per_px: float,
    eliminate_length_px: int,
    border_margin_px: int,
    merge_short_seg_px: int,
    threshold_of_nonlinear: float,
    blob_px: int,
    cut_max: int,
    cut_angle: float,
    pairing_angle_max: float,
    pairing_length_for_calc_angle: int,
    top_cut: int,
    post_eliminate_length_px: float,
):
    out_dir = ROOT / "data" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    img_bin = binarize(img_pre01, float(threshold_manual))
    img_for_skel = noize_elimination(img_bin, int(eliminate_length_px))
    img_skel = skeletonize(img_for_skel)

    graph = convert_to_graph(img_skel, int(border_margin_px))

    graph_nodes_merged = merge_nodes(graph, merge_short_seg_px)

    seg_before = int(len(graph_nodes_merged.segments))

    graph_kink = kink_cut(
        graph_nodes_merged,
        float(threshold_of_nonlinear),
        int(blob_px),
        int(cut_max),
        float(cut_angle),
    )
    seg_after = int(len(graph_kink.segments))
    split_count = int(max(0, seg_after - seg_before))

    graph_paired = pairing(
        graph_kink,
        float(pairing_angle_max),
        int(pairing_length_for_calc_angle),
    )
    pair_map = getattr(graph_paired, "pair_map", {}) or {}
    pairing_count = int(len(pair_map))

    fibers = measure_length(graph_paired, int(top_cut))

    fibers_csv = out_dir / f"{tag}__fibers.csv"
    used_cfg_json = out_dir / f"{tag}__used_config.json"

    used_cfg_payload = dict(
        um_per_px=float(um_per_px),
        threshold_otsu_recommended=float(threshold_otsu),
        threshold_manual=float(threshold_manual),
        eliminate_length_px=int(eliminate_length_px),
        border_margin_px=int(border_margin_px),
        threshold_of_nonlinear=float(threshold_of_nonlinear),
        blob_px=int(blob_px),
        cut_max=int(cut_max),
        cut_angle=float(cut_angle),
        pairing_angle_max=float(pairing_angle_max),
        pairing_length_for_calc_angle=int(pairing_length_for_calc_angle),
        top_cut=int(top_cut),
        post_eliminate_length_px=float(post_eliminate_length_px),
    )

    fibers_filtered = postprocess(
        fibers,
        float(post_eliminate_length_px),
        out_csv_path=fibers_csv,
        out_config_json_path=used_cfg_json,
        used_config=used_cfg_payload,
    )

    configure_draw_output(out_dir, tag)
    draw_separated_fiber_img(graph_paired, img_skel)

    paired_tif_path = out_dir / f"{tag}__paired_segments.tif"

    return dict(
        out_dir=str(out_dir),
        img_skel=img_skel,
        paired_tif=str(paired_tif_path),
        fibers_total=int(len(fibers)),
        fibers_used=int(len(fibers_filtered)),
        split_count=int(split_count),
        pairing_count=int(pairing_count),
        fibers_filtered=fibers_filtered,
    )

# ----------------------------
# Session state (for fixed layout)
# ----------------------------
if "file_id" not in st.session_state:
    st.session_state.file_id = None
if "threshold_manual" not in st.session_state:
    st.session_state.threshold_manual = None

# ----------------------------
# Sidebar: file uploader at top
# ----------------------------
st.sidebar.header("Input")
uploaded = st.sidebar.file_uploader(
    "Drag & drop here, or Browse file",
    type=["tif", "tiff", "png", "jpg", "jpeg"],
)

# ----------------------------
# Sidebar: parameters (all number inputs)
# ----------------------------
st.sidebar.header("Parameters (1st step: preprocess -> binarize)")

background_is_dark = bool(
    st.sidebar.toggle(
        "dark background",
        value=1 if bool(CFG.background_is_dark) else 0,
    )
)

um_per_px = float(
    st.sidebar.number_input(
        "**um_per_px:**  \n setting scale here",
        value=float(CFG.um_per_px),
        step=0.1,
        format="%.1f",
    )
)

blur_sigma_px = float(
    st.sidebar.number_input(
        "**blur_sigma_px:**  \n extent of blurring in generating background",
        value=float(CFG.blur_sigma_px),
        step=0.5,
        format="%.1f",
    )
)

threshold_otsu = float("nan")

# ----------------------------
# Sidebar: remaining parameters (render once; disabled until file is loaded)
# ----------------------------
sidebar_disabled = uploaded is None

threshold_manual = float(
    st.sidebar.number_input(
        "threshold_manual (0..1)",
        value=float(st.session_state.threshold_manual)
        if (st.session_state.threshold_manual is not None)
        else float(CFG.threshold),
        step=0.01,
        min_value=0.0,
        max_value=1.0,
        format="%.2f",
        disabled=sidebar_disabled,
    )
)
if not sidebar_disabled:
    st.session_state.threshold_manual = float(threshold_manual)

threshold_otsu_line_ph = st.sidebar.empty()
threshold_otsu_line_ph.write("threshold_otsu: -" if sidebar_disabled else "threshold_otsu (recommended): (computing...)")

st.sidebar.markdown("---")
run_lower_button = st.sidebar.button("Run analysis  \n (skeletonize -> results)", disabled=sidebar_disabled)
st.sidebar.markdown("---")
st.sidebar.header("Parameters (skeletonize -> results)")

eliminate_length_px = int(
    st.sidebar.number_input(
        "**eliminate_length_px:**  \n elminate object with this px*px area before skeletonize",
        value=int(CFG.eliminate_length_px),
        step=1,
        disabled=sidebar_disabled,
    )
)
border_margin_px = int(
    st.sidebar.number_input(
        "**border_margin_px:**  \n ignore object this px close to the edge",
        value=int(CFG.border_margin_px),
        step=1,
        disabled=sidebar_disabled,
    )
)
merge_short_seg_px = int(
    st.sidebar.number_input(
        "**merge_short_seg_px:**  \n merge nodes this px close to each other",
        value=int(CFG.merge_short_seg_px),
        step=1,
        disabled=sidebar_disabled,
    )
)

threshold_of_nonlinear = float(
    st.sidebar.number_input(
        "**threshold_of_nonlinear:**  \n L/D threshold for kink_cut condidate, where L: fiber length and D: edge distance",
        value=float(CFG.threshold_of_nonlinear),
        step=0.05 if sidebar_disabled else 0.01,
        format="%.2f",
        disabled=sidebar_disabled,
    )
)

blob_px = int(
    st.sidebar.number_input(
        "**(arm_length_px):**  \n arms length to calcuate kink angle",
        value=int(CFG.blob_px),
        step=1,
        disabled=sidebar_disabled,
    )
)
cut_max = int(
    st.sidebar.number_input(
        "**(cut_max):** \n max times of kink_cut applied",
        value=int(CFG.cut_max),
        step=1,
        disabled=sidebar_disabled,
    )
)
cut_angle = float(
    st.sidebar.number_input(
        "**cut_angle:**  \n cut object at a kink over this angle in degree",
        value=float(CFG.cut_angle),
        step=1.0 if sidebar_disabled else 0.5,
        format="%.0f" if sidebar_disabled else "%.2f",
        disabled=sidebar_disabled,
    )
)

pairing_angle_max = float(
    st.sidebar.number_input(
        "**pairing_angle_max:**  \n connect object joined below this angle in degree",
        value=float(CFG.pairing_angle_max),
        step=1.0 if sidebar_disabled else 0.5,
        format="%.0f" if sidebar_disabled else "%.6f",
        disabled=sidebar_disabled,
    )
)
pairing_length_for_calc_angle = int(
    st.sidebar.number_input(
        "**(pairing_length_for_calc_angle_px):**  \n arm length to calculate angle",
        value=int(CFG.pairing_length_for_calc_angle),
        step=1,
        disabled=sidebar_disabled,
    )
)

top_cut = int(
    st.sidebar.number_input(
        "**(top_cut_px):**  \n recude fiber length with this px at both end",
        value=int(CFG.top_cut),
        step=1,
        disabled=sidebar_disabled,
    )
)
post_eliminate_length_px = float(
    st.sidebar.number_input(
        "**post_eliminate_length_px:**  \n remove fibers less than this length from statistics",
        value=float(CFG.post_eliminate_length_px),
        step=5.0 if sidebar_disabled else 0.5,
        format="%.0f" if sidebar_disabled else "%.6f",
        disabled=sidebar_disabled,
    )
)

hist_min_um = float(
    st.sidebar.number_input(
        "**hist_range_min_um:**",
        value=float(CFG.hist_range[0]),
        step=10.0,
        format="%.0f" if sidebar_disabled else "%.6f",
        disabled=sidebar_disabled,
    )
)
hist_max_um = float(
    st.sidebar.number_input(
        "**hist_range_max_um:**",
        value=float(CFG.hist_range[1]),
        step=10.0,
        format="%.0f" if sidebar_disabled else "%.6f",
        disabled=sidebar_disabled,
    )
)
hist_bins = int(
    st.sidebar.number_input(
        "**hist_bins:**",
        value=int(CFG.hist_bins),
        step=1,
        disabled=sidebar_disabled,
    )
)

# ----------------------------
# Main: fixed layout placeholders (always rendered)
# ----------------------------
blank_gray = np.zeros((800, 800), dtype=np.uint8)
blank_rgb = np.zeros((800, 800, 3), dtype=np.uint8)

st.subheader("1st step: preprocess -> binarize")
u1, u2 = st.columns(2)
with u1:
    st.write("Preprocessed (background subtracted)")
    upper_pre_ph = st.image(blank_gray)
with u2:
    st.write("Binarized")
    upper_bin_ph = st.image(blank_gray)

st.markdown("---")

st.subheader("2nd step (skeletonize -> results)")
l1, l2 = st.columns(2)
with l1:
    st.write("Skeletonized")
    lower_skel_ph = st.image(blank_gray)
with l2:
    st.write("draw_separated_fiber_img output")
    lower_paired_ph = st.image(blank_rgb)

st.markdown("---")

st.subheader("Histogram (x: um, y: % of total length) with cumulative %")
hist_ph = st.empty()

st.write("R table + length-weighted mean (tab-separated)")
rtext_ph = st.empty()

st.write("Counts")
counts_ph = st.empty()

# ----------------------------
# If no file yet: keep placeholders and stop
# ----------------------------
if uploaded is None:
    rtext_ph.text("R0\t-\nR5\t-\nR10\t-\nR20\t-\nR50\t-\nR80\t-\nR90\t-\nR95\t-\nR100\t-\nMean(length-weighted)\t-")
    counts_ph.text("Total fibers (measured): 0\nUsed fibers (postprocess): 0\nSplit count: 0\nPairing count: 0")
    hist_ph.pyplot(plt.figure())
    plt.close("all")
    st.stop()

# ----------------------------
# With file: prepare temp path
# ----------------------------
file_bytes = uploaded.getvalue()
file_id = _file_id_from_bytes(file_bytes)
tag = Path(uploaded.name).stem

tmpdir = tempfile.TemporaryDirectory()
tmp_path = Path(tmpdir.name) / uploaded.name
tmp_path.write_bytes(file_bytes)

# new image -> reset caches and threshold default
if st.session_state.file_id != file_id:
    st.session_state.file_id = file_id
    st.session_state.upper_cache = None
    st.session_state.lower_cache = None
    st.session_state.threshold_manual = None

# ----------------------------
# Upper compute (auto-update)
# ----------------------------
img_pre01, threshold_otsu = compute_upper(str(tmp_path), background_is_dark, blur_sigma_px)

threshold_otsu_line_ph.write(f"threshold_otsu (recommended): {threshold_otsu:.3f}")

# Initialize threshold_manual from Otsu once per new image, then rerun so the sidebar reflects it.
if st.session_state.threshold_manual is None:
    st.session_state.threshold_manual = float(threshold_otsu - 0.02)
    st.rerun()

threshold_manual = float(st.session_state.threshold_manual)

img_bin = compute_binarized(img_pre01, threshold_manual)

upper_pre_ph.image(crop_center(img_pre01, 800), clamp=True)
upper_bin_ph.image(crop_center((img_bin.astype(np.uint8) * 255), 800))

# ----------------------------
# Lower compute only on button; placeholders exist from startup
# ----------------------------
if run_lower_button:
    st.session_state.lower_cache = compute_lower_and_save(
        img_pre01,
        threshold_manual,
        threshold_otsu,
        tag=tag,
        um_per_px=um_per_px,
        eliminate_length_px=eliminate_length_px,
        border_margin_px=border_margin_px,
        merge_short_seg_px=merge_short_seg_px,
        threshold_of_nonlinear=threshold_of_nonlinear,
        blob_px=blob_px,
        cut_max=cut_max,
        cut_angle=cut_angle,
        pairing_angle_max=pairing_angle_max,
        pairing_length_for_calc_angle=pairing_length_for_calc_angle,
        top_cut=top_cut,
        post_eliminate_length_px=post_eliminate_length_px,
    )

res = st.session_state.lower_cache

# ----------------------------
# Render lower outputs (or keep placeholders if not run yet)
# ----------------------------
if res is None:
    rtext_ph.text("R0\t-\nR5\t-\nR10\t-\nR20\t-\nR50\t-\nR80\t-\nR90\t-\nR95\t-\nR100\t-\nMean(length-weighted)\t-")
    counts_ph.text("Total fibers (measured): 0\nUsed fibers (postprocess): 0\nSplit count: 0\nPairing count: 0")

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()
    ax1.set_xlabel("Fiber length (um)")
    ax1.set_ylabel("Length-weighted %")
    ax2.set_ylabel("Cumulative %")
    ax2.set_ylim(0, 100)
    fig.tight_layout()
    hist_ph.pyplot(fig)
    plt.close(fig)

else:
    sk = res["img_skel"]
    lower_skel_ph.image(crop_center((sk.astype(np.uint8) * 255), 800))

    img_paired = iio.imread(res["paired_tif"])
    lower_paired_ph.image(crop_center(img_paired, 800))

    lengths_all = fibers_to_lengths_um(res["fibers_filtered"], um_per_px=float(um_per_px))
    r0, r1 = float(hist_min_um), float(hist_max_um)
    mask = (lengths_all >= r0) & (lengths_all <= r1)
    lengths = lengths_all[mask]
    weights = lengths.copy()

    hist, edges = np.histogram(lengths, bins=int(hist_bins), range=(r0, r1), weights=weights)
    total = float(hist.sum()) if hist.size else 0.0
    hist_pct = (hist / total * 100.0) if total > 0 else hist
    cum_pct = np.cumsum(hist_pct)

    centers = (edges[:-1] + edges[1:]) / 2.0
    widths = np.diff(edges)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()

    ax1.bar(centers, hist_pct, width=widths, align="center")
    ax1.set_xlabel("Fiber length (um)")
    ax1.set_ylabel("Length-weighted %")

    ax2.plot(centers, cum_pct, "r-")
    ax2.set_ylabel("Cumulative %")
    ax2.set_ylim(0, 100)

    fig.tight_layout()
    hist_ph.pyplot(fig)
    plt.close(fig)

    percentiles = [0, 5, 10, 50, 90, 95, 100]
    r_lines = []
    for p in percentiles:
        val = weighted_quantile(lengths, weights, p / 100.0)
        r_lines.append(f"R{p}\t{val:.1f}")

    if lengths.size > 0 and float(np.sum(lengths)) > 0:
        wmean = float(np.sum(lengths * lengths) / np.sum(lengths))
        r_lines.append(f"Mean(length-weighted)\t{wmean:.1f}")
    else:
        r_lines.append("Mean(length-weighted)\t-")

    rtext_ph.text("\n".join(r_lines))

    counts_ph.text(
        "\n".join(
            [
                f"Total fibers (measured): {res['fibers_total']}",
                f"Used fibers (postprocess): {res['fibers_used']}",
                f"Split count: {res['split_count']}",
                f"Pairing count: {res['pairing_count']}",
            ]
        )
    )

tmpdir.cleanup()
