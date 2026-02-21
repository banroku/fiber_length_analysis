# Path: config.py
# 役割: pipeline / app / scripts から参照される設定値を集約する

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # ---- scale ----
    um_per_px: float = 10.6  # 1 pixel が何 um か（表示やGUI入力の変換用）

    # ---- input ----
    background_is_dark: bool = False # False: 背景が明るく繊維が暗い / True: 背景が暗く繊維が明るい（input_imgで反転して以降は背景黒で統一）

    # ---- subtract_background ----
    blur_sigma_px: float = 7.0  # 背景推定の局所平均（masked_blurなど）で使うぼかしスケール

    # ---- binarize ----
    threshold: float = 0.25  # 手動閾値（0..1）。GUIでスライダー調整する想定。Otsuは推奨値表示のみ。

    # ---- noize_elimination ----
    eliminate_length_px: int = 5  # 縦・横がこれより小さい成分を除去（あなたの仕様）

    # ---- skeletonize ----
    # connectivity は 8 のみ実装（あなたの仕様）。ここに値を置いてもよいが固定なら不要。
    # connectivity: int = 8

    # ---- convert_to_graph ----
    border_margin_px: int = 3  # 画像端の繊維を測定対象から外す境界幅

    # ---- merge_nodes ----
    merge_short_seg_px: int = 3  # ノード間の距離がこれ以下なら、単一のノードとみなす

    # ---- kink_cut ----
    threshold_of_nonlinear: float = 1.20  # L/D がこれより大なら非直線セグメント候補
    blob_px: int = 7                    # 曲率評価のサンプル間隔
    cut_max: int = 2                     # キンク分割の最大回数（これ超えたらそのセグメントは削除）
    cut_angle: float = 15.0              # ここ(°)を超える折れ曲がりをキンクとみなす

    # ---- pairing ----
    pairing_angle_max: float = 25.0           # 折れ曲がり角(°)がこれ以下なら「直進」と判定してペア
    pairing_length_for_calc_angle: int = 12   # 角度計算のためノードから離れる距離(px)

    # ---- measure_length ----

    top_cut: int = 1  # スケルトナイズにより実際より長く両端をカットする

    # ---- postprocess ----
    post_eliminate_length_px: float = 5 # 最終的に短い繊維を除外する閾値(px)

    # ---- visualize result ----
    hist_range: tuple = (0, 2000) # ヒストグラムの表示範囲
    hist_bins: int = 30 # ヒストグラムの区間数

CFG = Config()
