import random
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt


from areas import divide_cross_section
from stones import (
    random_stone,
    snap_to_corner,
    snap_to_border,
)

def _remove_redundant_vertices(
    coords: List[Tuple[float, float]],
    tol: float = 1e-6
) -> List[Tuple[float, float]]:
    """
    Remove vertices that lie on purely horizontal or vertical straight segments.
    Does two-pass filtering: first horizontal, then vertical. Keeps endpoints of each segment.
    Works for closed polygons.

    Args:
        coords: List of (x, y) vertices, assumed in order.
        tol: Tolerance for coordinate equality to detect horizontal/vertical.

    Returns:
        Filtered list of vertices with redundant points removed.
    """
    def filter_direction(points: List[Tuple[float, float]], check_horizontal: bool) -> List[Tuple[float, float]]:
        if len(points) < 3:
            return points
        result: List[Tuple[float, float]] = []
        n = len(points)

        for i in range(n):
            prev_pt = points[i - 1]
            curr_pt = points[i]
            next_pt = points[(i + 1) % n]

            if check_horizontal:
                # Horizontal: y all equal within tol
                redundant = (
                    abs(curr_pt[1] - prev_pt[1]) < tol and
                    abs(next_pt[1] - curr_pt[1]) < tol
                )
            else:
                # Vertical: x all equal within tol
                redundant = (
                    abs(curr_pt[0] - prev_pt[0]) < tol and
                    abs(next_pt[0] - curr_pt[0]) < tol
                )

            if not redundant:
                result.append(curr_pt)
        # If filtering removed too many, return original
        return result if len(result) >= 3 else points

    # First pass: remove horizontal redundant vertices
    horiz_filtered = filter_direction(coords, check_horizontal=True)
    # Second pass: remove vertical redundant vertices
    final_filtered = filter_direction(horiz_filtered, check_horizontal=False)

    return final_filtered


def generate_cross_section(
    X: float = 0.06,
    Y: float = 0.04,
    n_rows: int = 2,
    min_div: int = 1,
    max_div: int = 3,
    min_width_frac: float = 0.1,
    min_height_frac: float = 0.1,
    TS_position: str = 'none',
    max_partition_attempts: int = 10000,
    sides: int = 10,
    K: float = 0.75,         # minimum stone area fraction
    border_thr: float = 0.1,
    corner_thr: float = 0.05,
    scale_prob: float = 0.3,
    min_scale: float = 0.95
) -> Tuple[
    List[List[float]],
    List[List[float]],
    List[List[Tuple[float, float]]],
    List[List[Tuple[float, float]]]
]:
    """
    Construct a 2D cross-section subdivided into stone‐filled cells.

    The rectangular domain of size (X × Y) is partitioned into `n_rows` horizontal
    bands, each further subdivided vertically into random cells. In each cell,
    a “stone” polygon is generated, optionally snapped to edges or corners,
    and returned both in its raw form and after cleanup (removal of colinear
    vertices). One special “through-stone” (TS) column may span the full height.

    Process overview:
      1. **Partition** the (X,Y) rectangle into `n_rows` bands and randomly
         subdivide each band horizontally into 1–`max_div` cells, obeying
         minimum width/height fractions.
      2. **(Optional) Through‐stone**: if `TS_position != 'none'`, find a column
         with matching widths in all rows and generate one tall stone spanning Y.
      3. **Stone generation**: for each cell, call `random_stone` to get a
         raw polygon of `sides` edges (with area ≥ K), scaled into the cell.
         Occasionally apply a random vertical squeeze to enhance variability.
      4. **Snapping**: push vertices close to cell borders or corners onto the
         exact cell boundary (thresholds: `border_thr`, `corner_thr`).
      5. **Cleanup**: translate snapped vertices into global coords and remove
         redundant straight‐line points.

    Parameters
    ----------
    X, Y : float
        Width and height of the overall cross-section domain (in meters).
    n_rows : int
        Number of horizontal stripes to divide into.
    min_div, max_div : int
        Minimum and maximum number of cells per stripe.
    min_width_frac, min_height_frac : float
        Minimum fraction of X (resp. Y) that each cell width (resp. stripe height)
        must occupy.
    TS_position : {'none', 'left', 'middle', 'right'}
        If not `'none'`, designates which column can host a single stone that
        traverses the full height of the domain.
    max_partition_attempts : int
        Maximum retries when trying to randomly partition stripes to satisfy
        the fraction constraints.
    sides : int
        Number of edges for each generated stone polygon (rough complexity).
    K : float
        Minimum required normalized area (0–1) for raw stones (passed to
        `random_stone`).
    border_thr : float
        Distance threshold (as fraction of stripe size) for snapping to cell
        edges.
    corner_thr : float
        Distance threshold for snapping to cell corners.
    scale_prob : float
        Probability of applying an extra vertical scale (0–1) to each non-TS stone.
    min_scale : float
        Minimum scale factor for that vertical squeeze (if applied).

    Returns
    -------
    Xlims : List[List[float]]
        For each of the `n_rows`, the list of cell widths in meters.
    Ylims : List[List[float]]
        For each of the `n_rows`, the list of stripe heights in meters.
    stones_raw : List[List[(x, y)]]
        For every cell (row × column), the raw list of stone-vertex coordinates
        in global (x,y) space, before snapping.
    stones_snap : List[List[(x, y)]]
        Corresponding list of snapped and cleaned vertex lists.
    """
    # --- 1) Partition the domain into subdivisions ---
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

    # --- 2) Pre-generate TS stone if requested ---
    ts_width = None
    xs_ts_local = ys_ts_local = None
    if TS_position != 'none':
        common_widths = set(Xlims[0])
        for wlist in Xlims[1:]:
            common_widths &= set(wlist)
        if common_widths:
            ts_width = common_widths.pop()
            xs_ts_local, ys_ts_local = random_stone(
                width=ts_width,
                height=Y,
                sides=int(sides * 1.5),
                min_scale=1,
            )

    # --- 3) Generate stones per cell and apply snapping rules ---
    y_offset = 0.0
    for row_idx, widths in enumerate(Xlims):
        stripe_heights = Ylims[row_idx]
        heights = (
            stripe_heights
            if len(stripe_heights) == len(widths)
            else [stripe_heights[0]] * len(widths)
        )
        x_offset = 0.0
        ts_col = widths.index(ts_width) if ts_width in widths else None

        for col_idx, (w, h) in enumerate(zip(widths, heights)):
            # Skip duplicate TS cells beyond the first row
            if ts_col is not None and col_idx == ts_col and row_idx > 0:
                x_offset += w
                continue

            # --- Raw stone geometry ---
            if ts_col is not None and col_idx == ts_col and row_idx == 0:
                xs_cell, ys_cell = xs_ts_local, ys_ts_local
            else:
                xs_cell, ys_cell = random_stone(
                    width=w, height=h, sides=sides, min_scale=min_scale
                )
                
            # Random vertical scaling for non-TS stones
            is_ts = (ts_col is not None and col_idx == ts_col and row_idx == 0)
            if not is_ts and random.random() < scale_prob:
                scale_y = random.uniform(0.5, 0.8)
                # choose horizontal wall pivot: bottom if cell center below mid-domain, else top
                cell_center_y = y_offset + h/2
                pivot_local_y = 0 if cell_center_y < Y/2 else h
                ys_cell = [(y - pivot_local_y) * scale_y + pivot_local_y for y in ys_cell]

            raw_coords = [(x + x_offset, y + y_offset)
                          for x, y in zip(xs_cell, ys_cell)]
            stones_raw.append(raw_coords)

            # --- Snapping ---
            xs2 = np.array(xs_cell)
            ys2 = np.array(ys_cell)
            if ts_col is not None and col_idx == ts_col and row_idx == 0:
                xs2, ys2 = snap_to_border(xs2, ys2, 'bottom', ts_width, Y, border_thr/n_rows)
                xs2, ys2 = snap_to_border(xs2, ys2, 'top', ts_width, Y, border_thr/n_rows)
                num_cols = len(widths)
                if col_idx == 0:
                    xs2, ys2 = snap_to_border(xs2, ys2, 'left', ts_width, Y, border_thr/n_rows)
                    xs2, ys2 = snap_to_corner(xs2, ys2, 'bottom-left', ts_width, Y, corner_thr)
                    xs2, ys2 = snap_to_corner(xs2, ys2, 'top-left', ts_width, Y, corner_thr)
                elif col_idx == num_cols - 1:
                    xs2, ys2 = snap_to_border(xs2, ys2, 'right', ts_width, Y, border_thr/n_rows)
                    xs2, ys2 = snap_to_corner(xs2, ys2, 'bottom-right', ts_width, Y, 0)
                    xs2, ys2 = snap_to_corner(xs2, ys2, 'top-right', ts_width, Y, 0)
            else:
                if row_idx == 0 and col_idx == 0:
                    xs2, ys2 = snap_to_corner(xs2, ys2, 'bottom-left', w, h, corner_thr)
                elif row_idx == 0 and col_idx == len(widths) - 1:
                    xs2, ys2 = snap_to_corner(xs2, ys2, 'bottom-right', w, h, corner_thr)
                elif row_idx == n_rows - 1 and col_idx == 0:
                    xs2, ys2 = snap_to_corner(xs2, ys2, 'top-left', w, h, corner_thr)
                elif row_idx == n_rows - 1 and col_idx == len(widths) - 1:
                    xs2, ys2 = snap_to_corner(xs2, ys2, 'top-right', w, h, corner_thr)
                elif row_idx == 0:
                    xs2, ys2 = snap_to_border(xs2, ys2, 'bottom', w, h, border_thr)
                elif row_idx == n_rows - 1:
                    xs2, ys2 = snap_to_border(xs2, ys2, 'top', w, h, border_thr)

            # Translate snapped coords and remove redundant colinear vertices
            snapped_global = [(x + x_offset, y + y_offset)
                              for x, y in zip(xs2, ys2)]
            snapped_clean = _remove_redundant_vertices(snapped_global)
            stones_snap.append(snapped_clean)

            x_offset += w
        y_offset += heights[0]

    return Xlims, Ylims, stones_raw, stones_snap


def draw_layout(ax, Xlims, Ylims, stones, X, Y):
    """
    Draw the cell partitions and stone polygons on a Matplotlib axis.

    - Partition lines in black.
    - Stones as polygons with blue vertex markers.
    """
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
        ax.scatter(xs[:-1], ys[:-1], c='blue', s=20, zorder=5)

    ax.set_xlim(0, X)
    ax.set_ylim(0, Y)
    ax.set_aspect('equal')
    ax.axis('off')


def main():
    """
    Example usage: generate multiple pairs of raw vs snapped layouts.
    """
    X, Y = 0.12, 0.04
    n_rows = 2
    min_div, max_div = 3, 6
    min_width_frac, min_height_frac = 0.1, 0.2
    TS_position = 'center'
    sides = 15  # base vertex count for standard stones
    n_pairs = 4
    max_attempts = n_pairs * 10
    K = 0.0
    corner_thr = 0.2
    border_thr = 0.3

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
        print(f" !! Only generated {row} out of {n_pairs} valid examples.")
        for r in range(row, n_pairs):
            axes[r, 0].axis('off')
            axes[r, 1].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
