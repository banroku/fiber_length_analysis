# app/app.py
import sys
from pathlib import Path
import tempfile
import hashlib

import numpy as np
import streamlit as st
import imageio.v3 as iio
import matplotlib.pyplot as plt
from io import BytesIO

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from fiberlen.config import CFG, Config
from fiberlen.load_img import load_img
from fiberlen.config_io import save_cfg_json, load_cfg_json
from fiberlen.subtract_background import subtract_background
from fiberlen.calc_otsu_threshold import calc_otsu_threshold
from fiberlen.binarize import binarize
from fiberlen.noize_elimination import noize_elimination
from fiberlen.skeletonize import skeletonize
from fiberlen.trim_graph import trim_graph 
from fiberlen.merge_nodes import merge_nodes
from fiberlen.convert_to_graph import convert_to_graph
from fiberlen.kink_cut import kink_cut
from fiberlen.pairing import pairing
from fiberlen.measure_length import measure_length
from fiberlen.postprocess import postprocess
from fiberlen.draw_separated_fiber_img import draw_separated_fiber_img
from fiberlen.save_fiber_list_csv import save_fiber_list_csv

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


def compute_upper(
    img_path: str,
    cfg: Config,
):
    img_raw01 = load_img(img_path, cfg.background_is_dark)
    img_pre01 = subtract_background(img_raw01, cfg.blur_sigma_px)
    img_bin = binarize(img_pre01, cfg.threshold)
    thr_otsu = float(calc_otsu_threshold(img_pre01))

    return img_raw01, img_pre01, img_bin, thr_otsu


def compute_middle_and_save(
    img_bin: np.ndarray,
    cfg: Config,
):
    img_for_skel = noize_elimination(img_bin, cfg.eliminate_length_px)
    img_skel = skeletonize(img_for_skel)

    return dict(
            img_for_skel=img_for_skel,
            img_skel=img_skel,
    )


def compute_lower_and_save(
    img_skel: np.ndarray,
    tag: str,
    cfg: Config,
):
    out_dir = ROOT / "data" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    graph = convert_to_graph(img_skel, cfg.border_margin_px)
    graph_trimmed = trim_graph(graph, cfg.trim_length_px)

    graph_nodes_merged = merge_nodes(graph_trimmed, cfg.merge_short_seg_px)

    seg_before = int(len(graph_nodes_merged.segments))

    graph_kink = kink_cut(
        graph_nodes_merged,
        cfg.threshold_of_nonlinear,
        cfg.cut_max,
        cfg.cut_angle
    )
    seg_after = int(len(graph_kink.segments))
    split_count = int(max(0, seg_after - seg_before))

    graph_paired = pairing(
        graph_kink,
        cfg.pairing_angle_max,
        cfg.pairing_length_for_calc_angle
    )
    pair_map = getattr(graph_paired, "pair_map", {}) or {}
    pairing_count = int(len(pair_map))

    fibers = measure_length(graph_paired, cfg.top_cut)

    fibers_csv = out_dir / f"{tag}__fibers.csv"
    used_cfg_json = out_dir / f"{tag}__used_config.json"

    fibers_filtered = postprocess(
        fibers,
        cfg.post_eliminate_length_um,
        cfg.um_per_px
    )

    img_labeled = draw_separated_fiber_img(graph_paired, img_skel)

    paired_tif_path = out_dir / f"{tag}__paired_segments.tif"

    return dict(
        img_skel=img_skel,
        img_labeled=img_labeled,
        fibers_total=int(len(fibers)),
        fibers_used=int(len(fibers_filtered)),
        split_count=int(split_count),
        pairing_count=int(pairing_count),
        fibers_filtered=fibers_filtered,
    )

#cfg = CFG
# ----------------------------
# Session state (for fixed layout)
# ----------------------------
if "file_id" not in st.session_state:
    st.session_state.file_id = None
if "threshold" not in st.session_state:
    st.session_state.threshold = None
if "cfg" not in st.session_state:
    st.session_state.cfg = CFG 


# ----------------------------
# Sidebar: file uploader at top
# ----------------------------
st.sidebar.header("Data")
uploaded = st.sidebar.file_uploader(
    "load image",
    type=["tif", "tiff", "png", "jpg", "jpeg"],
)
sidebar_disabled = uploaded is None


# ----------------------------
# Sidebar: config loader
# ----------------------------
uploaded_cfg = st.sidebar.file_uploader(
    "load settings (.json)",
    type="json",
)

if uploaded_cfg is not None:
    # 一時ファイルに保存
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
        tmp.write(uploaded_cfg.read())
        tmp_path = tmp.name

    # dataclass として復元
    loaded_cfg = load_cfg_json(Config, tmp_path)

    # session_state に差し替え
    st.session_state.cfg = loaded_cfg
    #cfg = loaded_cfg
   
    # cfgに代入したら、uploaded_cfgのフラグは落とす
    uploaded_cfg = None

cfg = st.session_state.cfg


# ----------------------------
# Sidebar: parameters (all number inputs)
# ----------------------------
# st.sidebar.markdown("---")
# st.sidebar.subheader("Load image & binarize")

cfg.background_is_dark = bool(
    st.sidebar.toggle(
        "dark background",
        value=1 if bool(st.session_state.cfg.background_is_dark) else 0,
    )
)

cfg.blur_sigma_px = float(
    st.sidebar.number_input(
        "**blur_sigma_px:**  \n blurring in generating background",
        value=float(st.session_state.cfg.blur_sigma_px),
        step=0.5,
        format="%.1f",
    )
)

cfg.threshold_otsu = float("nan")

cfg.threshold = float(
    st.sidebar.number_input(
        "**threshold:** (0 ot 1)",
        value=float(st.session_state.threshold)
        if (st.session_state.threshold is not None)
        else float(st.session_state.cfg.threshold),
        step=0.01,
        min_value=0.0,
        max_value=1.0,
        format="%.2f",
    )
)

threshold_otsu_line_ph = st.sidebar.empty()
threshold_otsu_line_ph.write("threshold_otsu (recommended): (computing...)")


st.sidebar.markdown("---")
st.sidebar.subheader("Eliminate noise and skeletonise")

run_middle_button = st.sidebar.button("Eliminate_noise -> skeletonize", disabled=sidebar_disabled)

cfg.eliminate_length_px = int(
    st.sidebar.number_input(
        "**eliminate_length_px:**  \n elminate object with this px*px area before skeletonize",
        value=int(st.session_state.cfg.eliminate_length_px),
        step=1,
    )
)

st.sidebar.markdown("---")
st.sidebar.subheader("Generate graph and Histrogram")

run_lower_button = st.sidebar.button("Generate", disabled=sidebar_disabled)

cfg.um_per_px = float(
    st.sidebar.number_input(
        "**um_per_px:**  \n setting scale here",
        value=float(st.session_state.cfg.um_per_px),
        step=0.1,
        format="%.2f",
    )
)
    
cfg.post_eliminate_length_um = float(
    st.sidebar.number_input(
        "**post_eliminate_length_um**  \n remove fibers less than this length from statistics",
        value=float(st.session_state.cfg.post_eliminate_length_um),
        step=5.0 if sidebar_disabled else 5.0,
        format="%.0f" if sidebar_disabled else "%.0f",
    )
)
    
with st.sidebar.expander("Graph and Histrogram settings", expanded=False):
    cfg.hist_min_um = float(
        st.number_input(
            "**hist_min_um:**",
            value=float(st.session_state.cfg.hist_min_um),
            step=10.0,
            format="%.0f" if sidebar_disabled else "%.0f",
        )
    )
    
    cfg.hist_max_um = float(
        st.number_input(
            "**hist_max_um:**",
            value=float(st.session_state.cfg.hist_max_um),
            step=10.0,
            format="%.0f" if sidebar_disabled else "%.0f",
        )
    )
    cfg.hist_bins = int(
        st.number_input(
            "**hist_bins:**",
            value=int(st.session_state.cfg.hist_bins),
            step=1,
        )
    )



# ----------------------------
# Sidebar: other parameters
# ----------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("Other analysis settings")

with st.sidebar.expander("convert_to_graph.py", expanded=False):
    cfg.border_margin_px = int(
        st.number_input(
            "**border_margin_px:**  \n ignore object this px close to the edge",
            value=int(st.session_state.cfg.border_margin_px),
            step=1,
        )
    )

with st.sidebar.expander("trim_graph.py", expanded=False):
    cfg.trim_length_px = int(
        st.number_input(
            "**trim_length_px:**  \n trim short branch less than this px",
            value=int(st.session_state.cfg.trim_length_px),
            step=1,
        )
    )

with st.sidebar.expander("merge_nodes.py", expanded=False):
    cfg.merge_short_seg_px = int(
        st.number_input(
            "**merge_short_seg_px:**  \n merge nodes this px close to each other",
            value=int(st.session_state.cfg.merge_short_seg_px),
            step=1,
        )
    )

with st.sidebar.expander("kink_cut.py", expanded=False):
    cfg.threshold_of_nonlinear = float(
        st.number_input(
            "**threshold_of_nonlinear:**  \n L/D threshold for kink_cut condidate, where L: fiber length and D: edge distance",
            value=float(st.session_state.cfg.threshold_of_nonlinear),
            step=0.05 if sidebar_disabled else 0.01,
            format="%.2f",
        )
    )
    
 #   cfg.blob_px = int(
 #       st.number_input(
 #           "**(arm_length_px):**  \n arms length to calcuate kink angle",
 #           value=int(st.session_state.cfg.blob_px),
 #           step=1,
 #       )
 #   )
 #   
    cfg.cut_max = int(
        st.number_input(
            "**(cut_max):** \n max times of kink_cut applied",
            value=int(st.session_state.cfg.cut_max),
            step=1,
        )
    )
    
    cfg.cut_angle = float(
        st.number_input(
            "**cut_angle:**  \n cut object at a kink over this angle in degree",
            value=float(st.session_state.cfg.cut_angle),
            step=1.0 if sidebar_disabled else 1.0,
            format="%.0f" if sidebar_disabled else "%.0f",
        )
    )


with st.sidebar.expander("pairing.py", expanded=False):
    cfg.pairing_angle_max = float(
        st.number_input(
            "**pairing_angle_max:**  \n connect object joined below this angle in degree",
            value=float(st.session_state.cfg.pairing_angle_max),
            step=1.0 if sidebar_disabled else 1.0,
            format="%.0f" if sidebar_disabled else "%.0f",
        )
    )
    
    cfg.pairing_length_for_calc_angle = int(
        st.number_input(
            "**(pairing_length_for_calc_angle_px):**  \n arm length to calculate angle",
            value=int(st.session_state.cfg.pairing_length_for_calc_angle),
            step=1,
        )
    )

with st.sidebar.expander("measure_length.py", expanded=False):
    cfg.top_cut = int(
        st.number_input(
            "**(top_cut_px):**  \n recude fiber length with this px at both end",
            value=int(st.session_state.cfg.top_cut),
            step=1,
        )
    )

# ----------------------------
# Sidebar: config downloader
# ----------------------------
st.sidebar.markdown("---")
st.sidebar.write("Save settings")

with tempfile.TemporaryDirectory() as td:
    tmp_path = Path(td) / "cfg.json"
    save_cfg_json(cfg, str(tmp_path))

    json_bytes = tmp_path.read_bytes()

st.sidebar.download_button(
    label="save settings as .json",
    data=json_bytes,
    file_name = "fiber_length_analysis_config.json",
    mime="application/json",
)



# ----------------------------
# Main: fixed layout placeholders (always rendered)
# ----------------------------
blank_gray = np.zeros((800, 800), dtype=np.uint8)
blank_rgb = np.zeros((800, 800, 3), dtype=np.uint8)

st.subheader("Preprocess")
u1, u2, u3 = st.columns(3)

with u1:
    st.write("BG subtracted")
    disp_img_01 = st.image(blank_gray)
with u2:
    st.write("Binarized")
    disp_img_02 = st.image(blank_gray)
with u3:
    st.write("Noise eliminated")
    disp_img_03 = st.image(blank_gray)


st.markdown("---")
st.subheader("Alalysis result with color label")

l1, l2 = st.columns(2)
with l1:
    st.write("Skeletonized")
    disp_img_04 = st.image(blank_gray)
with l2:
    st.write("Identified fibers")
    disp_img_05 = st.image(blank_rgb)


st.markdown("---")
st.subheader("Histogram (length-weighted)")

hist_ph = st.empty()

st.write("R table (length-weighted)")
rtext_ph = st.empty()

st.write("Counts")
counts_ph = st.empty()


# ----------------------------
# If no file yet: keep placeholders and stop
# ----------------------------
if uploaded is None: st.stop()


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
    st.session_state.threshold = None
    st.session_state.upper_cache = None
    st.session_state.lower_cache = None
    st.session_state.middle_cache = None

# ----------------------------
# Upper compute (auto-update)
# ----------------------------
img_raw01, img_pre01, img_bin, threshold_otsu  = compute_upper(str(tmp_path), cfg)

threshold_otsu_line_ph.write(f"threshold_otsu (recommended): {threshold_otsu:.3f}")

# # Initialize threshold from Otsu once per new image, then rerun so the sidebar reflects it.
# if st.session_state.threshold is None:
#     st.session_state.threshold = float(threshold_otsu - 0.02)
#     st.rerun()

img_bin = binarize(img_pre01, cfg.threshold)

disp_img_01.image(crop_center(img_pre01, 800), clamp=True)
disp_img_02.image(crop_center((img_bin.astype(np.uint8) * 255), 800))

# ----------------------------
# compute middle only on button; placeholders exist from startup
# ----------------------------
if run_middle_button:
    st.session_state.middle_cache = compute_middle_and_save(img_bin, cfg)

res_middle = st.session_state.middle_cache

# ----------------------------
# compute lower only on button; placeholders exist from startup
# ----------------------------
if run_lower_button:
    st.session_state.lower_cache = compute_lower_and_save(
        res_middle["img_skel"],
        tag=tag,
        cfg=cfg,
    )

res = st.session_state.lower_cache

# ----------------------------
# Render middle outputs (or keep placeholders if not run yet)
# ----------------------------
if res_middle is None:
    disp_img_03.image(blank_gray)
    disp_img_04.image(blank_gray)
    
else:
    img_for_skel = res_middle["img_for_skel"]
    disp_img_03.image(crop_center((img_for_skel.astype(np.uint8) * 255), 800))

    img_skel = res_middle["img_skel"]
    disp_img_04.image(crop_center((img_skel.astype(np.uint8) * 255), 800))

# ----------------------------
# Render lower outputs (or keep placeholders if not run yet)
# ----------------------------
if not res is None:
    disp_img_05.image(crop_center(res["img_labeled"], 800))

    lengths_all = fibers_to_lengths_um(res["fibers_filtered"], um_per_px=float(cfg.um_per_px))
    r0, r1 = float(cfg.hist_min_um), float(cfg.hist_max_um)
    mask = (lengths_all >= r0) & (lengths_all <= r1)
    lengths = lengths_all[mask]
    weights = lengths.copy()

    hist, edges = np.histogram(lengths, bins=int(cfg.hist_bins), range=(r0, r1), weights=weights)
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
    ax1.set_ylabel("Length-weighted frequency (%)")

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


st.markdown("---")
st.markdown("### Fiber List CSV")

cfg = st.session_state.cfg

if res is not None and len(res["fibers_filtered"]) > 0:
    csv_text = save_fiber_list_csv(
        fibers=res["fibers_filtered"],
        filename=tag,
        um_per_px=cfg.um_per_px,
    )

    st.download_button(
        label="Fiber list をCSVで保存",
        data=csv_text,
        file_name=f"{tag}_fiber_list.csv",
        mime="text/csv",
    )
else:
    st.caption("fibers が存在しません")


if res is not None:
    bio = BytesIO()
    iio.imwrite(bio, res["img_labeled"], extension=".png")
    
    st.download_button(
        label="ラベル画像を保存",
        data=bio.getvalue(),
        file_name=f"{tag}_labeled.png",
        mime="image/png",
    )

tmpdir.cleanup()
