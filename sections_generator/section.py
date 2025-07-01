import random
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from areas import divide_cross_section
from stones import (
    random_stone,
    snap_to_corner,
    snap_to_border,
    clip_polygon_to_rect,
)


def generate_cross_section(
    X: float = 0.6,
    Y: float = 0.4,
    n_rows: int = 2,
    min_div: int = 1,
    max_div: int = 3,
    min_width_frac: float = 0.1,
    min_height_frac: float = 0.1,
    TS_position: str = 'none',
    max_partition_attempts: int = 10000,
    sides: int = 20,
    K: float = 0.7,         # minimum stone area fraction
    border_thr: float = 0.01,
    corner_thr: float = 0.0,
) -> Tuple[
    List[List[float]],
    List[List[float]],
    List[List[Tuple[float, float]]],
    List[List[Tuple[float, float]]]
]:
    """
    Generate a cross-section layout with raw and snapped stones.
    Handles Through-Stone (TS) by clipping and then snapping.

    Returns:
        Xlims, Ylims, stones_raw, stones_snap
    """
    # 1) Partition the area
    Xlims, Ylims = divide_cross_section(
        X=X, Y=Y,
        n_rows=n_rows,
        min_div=min_div, max_div=max_div,
        min_width_frac=min_width_frac,
        min_height_frac=min_height_frac,
        TS_position=TS_position,
        max_partition_attempts=max_partition_attempts,
    )

    stones_raw: List[List[Tuple[float, float]]] = []
    stones_snap: List[List[Tuple[float, float]]] = []

    # 2) If TS present, determine TS width
    ts_width = None
    if TS_position != 'none':
        common = set(Xlims[0])
        for w in Xlims[1:]:
            common &= set(w)
        if common:
            ts_width = common.pop()
            xs_ts_local, ys_ts_local = random_stone(
                sides=sides, width=ts_width, height=Y, K=K
            )

    # 3) Build stones per stripe
    y_offset = 0.0
    for row_idx, widths in enumerate(Xlims):
        stripe_hs = Ylims[row_idx]
        heights = (
            stripe_hs
            if len(stripe_hs) == len(widths)
            else [stripe_hs[0]] * len(widths)
        )
        x_offset = 0.0
        # find TS column index
        ts_idx = widths.index(ts_width) if ts_width in widths else None
        x_offset_ts = sum(widths[:ts_idx]) if ts_idx is not None else None

        for col_idx, (w, h) in enumerate(zip(widths, heights)):
            # --- RAW ---
            if ts_idx is not None and col_idx == ts_idx:
                # clip TS to its cell first
                y_min, y_max = y_offset, y_offset + h
                xs_clip, ys_clip = clip_polygon_to_rect(
                    xs_ts_local + x_offset_ts, ys_ts_local,
                    x_offset_ts, x_offset_ts + ts_width,
                    y_min, y_max
                )
                coords_raw = list(zip(xs_clip, ys_clip))
            else:
                xs_cell, ys_cell = random_stone(sides=sides, width=w, height=h, K=K)
                coords_raw = [
                    (x + x_offset, y + y_offset)
                    for x, y in zip(xs_cell, ys_cell)
                ]
            stones_raw.append(coords_raw)

            # --- SNAPPED ---
            if ts_idx is not None and col_idx == ts_idx:
                # clip → local coords
                y_min, y_max = y_offset, y_offset + h
                xs_clip, ys_clip = clip_polygon_to_rect(
                    xs_ts_local + x_offset_ts, ys_ts_local,
                    x_offset_ts, x_offset_ts + ts_width,
                    y_min, y_max
                )
                xs_loc = xs_clip - x_offset
                ys_loc = ys_clip - y_offset

                # then snap locally
                if row_idx == 0 and col_idx == 0:
                    xs2_loc, ys2_loc = snap_to_corner(
                        xs_loc, ys_loc, 'bottom-left', w, h, corner_thr
                    )
                elif row_idx == 0 and col_idx == len(widths) - 1:
                    xs2_loc, ys2_loc = snap_to_corner(
                        xs_loc, ys_loc, 'bottom-right', w, h, corner_thr
                    )
                elif row_idx == n_rows - 1 and col_idx == 0:
                    xs2_loc, ys2_loc = snap_to_corner(
                        xs_loc, ys_loc, 'top-left', w, h, corner_thr
                    )
                elif row_idx == n_rows - 1 and col_idx == len(widths) - 1:
                    xs2_loc, ys2_loc = snap_to_corner(
                        xs_loc, ys_loc, 'top-right', w, h, corner_thr
                    )
                elif row_idx == 0:
                    xs2_loc, ys2_loc = snap_to_border(
                        xs_loc, ys_loc, 'bottom', w, h, border_thr
                    )
                elif row_idx == n_rows - 1:
                    xs2_loc, ys2_loc = snap_to_border(
                        xs_loc, ys_loc, 'top', w, h, border_thr
                    )
                else:
                    xs2_loc, ys2_loc = xs_loc, ys_loc

                coords_snap = [
                    (x + x_offset, y + y_offset)
                    for x, y in zip(xs2_loc, ys2_loc)
                ]
            else:
                # non-TS: just generate & snap
                xs2, ys2 = random_stone(sides=sides, width=w, height=h, K=K)
                if row_idx == 0 and col_idx == 0:
                    xs2, ys2 = snap_to_corner(
                        xs2, ys2, 'bottom-left', w, h, corner_thr
                    )
                elif row_idx == 0 and col_idx == len(widths) - 1:
                    xs2, ys2 = snap_to_corner(
                        xs2, ys2, 'bottom-right', w, h, corner_thr
                    )
                elif row_idx == n_rows - 1 and col_idx == 0:
                    xs2, ys2 = snap_to_corner(
                        xs2, ys2, 'top-left', w, h, corner_thr
                    )
                elif row_idx == n_rows - 1 and col_idx == len(widths) - 1:
                    xs2, ys2 = snap_to_corner(
                        xs2, ys2, 'top-right', w, h, corner_thr
                    )
                elif row_idx == 0:
                    xs2, ys2 = snap_to_border(
                        xs2, ys2, 'bottom', w, h, border_thr
                    )
                elif row_idx == n_rows - 1:
                    xs2, ys2 = snap_to_border(
                        xs2, ys2, 'top', w, h, border_thr
                    )
                coords_snap = [
                    (x + x_offset, y + y_offset)
                    for x, y in zip(xs2, ys2)
                ]

            stones_snap.append(coords_snap)
            x_offset += w

        y_offset += heights[0]

    return Xlims, Ylims, stones_raw, stones_snap


def draw_layout(ax, Xlims, Ylims, stones, X, Y):
    y0 = 0.0
    for stripe in Ylims:
        ax.plot([0, X], [y0, y0], 'k-')
        y0 += stripe[0]
    ax.plot([0, X], [Y, Y], 'k-')

    y0 = 0.0
    for row_idx, widths in enumerate(Xlims):
        h = Ylims[row_idx][0]
        x0 = 0.0
        for w in widths:
            ax.plot([x0, x0], [y0, y0 + h], 'k-')
            x0 += w
        ax.plot([X, X], [y0, y0 + h], 'k-')
        y0 += h

    for poly in stones:
        if not poly:
            continue
        pts = poly + [poly[0]]
        xs, ys = zip(*pts)
        ax.plot(xs, ys, 'k-')

    ax.set_xlim(0, X)
    ax.set_ylim(0, Y)
    ax.set_aspect('equal')
    ax.axis('off')


def main():
    X, Y = 1, 0.5
    n_rows = 2
    min_div, max_div = 3, 4
    min_width_frac, min_height_frac = 0.05, 0.2
    TS_position = 'center'
    sides = 16
    n_pairs = 4
    max_attempts = n_pairs * 10
    K = 0.2,
    corner_thr = 0.0
    border_thr = 0.1

    fig, axes = plt.subplots(n_pairs, 2, figsize=(10, n_pairs * 2.5))

    row = 0
    attempts = 0
    while row < n_pairs and attempts < max_attempts:
        attempts += 1
        try:
            Xlims, Ylims, raw, snap = generate_cross_section(
                X, Y, n_rows,
                min_div, max_div,
                min_width_frac, min_height_frac,
                TS_position,
                sides=sides,
                corner_thr=corner_thr,
                border_thr=border_thr,
                K=K
            )
        except ValueError as e:
            print(f"[{row}] Skipping due to ValueError: {e}")
            continue

        draw_layout(axes[row, 0], Xlims, Ylims, raw, X, Y)
        axes[row, 0].set_title('Raw')
        draw_layout(axes[row, 1], Xlims, Ylims, snap, X, Y)
        axes[row, 1].set_title('Snapped')
        row += 1

    if row < n_pairs:
        print(f"⚠️ Only generated {row} out of {n_pairs} valid examples.")
        for r in range(row, n_pairs):
            axes[r, 0].axis('off')
            axes[r, 1].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
