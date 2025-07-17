import random
import numpy as np

# HELPER FUNCTIONS
def _sample_sorted_coords(sides: int) -> list[int]:
    """
    Sample `sides` unique integers from 0..99 and return them sorted.
    """
    return sorted(random.sample(range(100), sides))


def _split_chain(coords: list[int]) -> tuple[list[int], list[int]]:
    """
    Given sorted coords, split the interior points into two random chains,
    each including the first and last point.
    """
    interior = coords[1:-1]
    random.shuffle(interior)
    half = len(interior) // 2
    a, b = sorted(interior[:half]), sorted(interior[half:])
    return [coords[0]] + a + [coords[-1]], [coords[0]] + b + [coords[-1]]

# MAIN FUNCTIONS
def random_stone(
    width: float = 0.35,
    height: float = 0.25,
    K: float = 0.0,
    sides: int = 20,
    min_scale: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a randomized 2D polygon resembling a natural stone silhouette.

    This function builds a closed polygon by:
    1. Sampling and sorting random x– and y–coordinates to form edge vectors.
    2. Converting those vectors into a connected loop of vertices.
    3. Normalizing to unit square and checking that the polygon’s area ≥ K.
       If area is too small, up to 100 retries are attempted.
    4. Applying independent random scaling factors (between min_scale and 1.0)
       in each quadrant to introduce irregularity.
    5. Scaling the shape to fit inside a box of size (width × height).
    6. Randomly translating it to one of five anchor positions within that box.

    Parameters
    ----------
    width : float, optional
        Total width of the bounding box in which the stone must lie (default 0.35).
    height : float, optional
        Total height of the bounding box (default 0.25).
    K : float, optional
        Minimum acceptable area fraction of the unit square (0 ≤ K < 1). Shapes
        with normalized area below K are discarded and retried (default 0.0).
    sides : int, optional
        Target number of edges (and thus vertices) for the raw polygon. Values
        between 10 and 25 produce more/less complex shapes (default 20).
    min_scale : float, optional
        Lower bound on quadrant scaling factors; controls how “jagged” the stone
        silhouette can be (0 < min_scale ≤ 1.0, default 0.95).

    Returns
    -------
    xs : np.ndarray of float
        X-coordinates of the final polygon vertices, scaled to [0, width].
    ys : np.ndarray of float
        Y-coordinates of the final polygon vertices, scaled to [0, height].
    """
    for _ in range(100):
        # 1) build raw polygon
        x_coords = _sample_sorted_coords(sides)
        y_coords = _sample_sorted_coords(sides)
        xa, xb = _split_chain(x_coords)
        ya, yb = _split_chain(y_coords)
        xb.reverse(); yb.reverse()
        x_vecs = [(xa[i], xa[i+1]) for i in range(len(xa)-1)] + [(xb[i], xb[i+1]) for i in range(len(xb)-1)]
        y_vecs = [(ya[i], ya[i+1]) for i in range(len(ya)-1)] + [(yb[i], yb[i+1]) for i in range(len(yb)-1)]
        random.shuffle(y_vecs)

        edges = []
        for (x0, x1), (y0, y1) in zip(x_vecs, y_vecs):
            dx, dy = x1 - x0, y1 - y0
            ang = np.arctan2(dy, dx)
            ln = np.hypot(dx, dy)
            edges.append((ang, ln))
        edges.sort(key=lambda e: e[0])

        xs, ys = [0.0], [0.0]
        for ang, ln in edges:
            xs.append(xs[-1] + ln * np.cos(ang))
            ys.append(ys[-1] + ln * np.sin(ang))
        xs_np = np.array(xs[:-1])
        ys_np = np.array(ys[:-1])

        # 2) normalize 
        xs_u = (xs_np - xs_np.min()) / (np.ptp(xs_np) + 1e-8)
        ys_u = (ys_np - ys_np.min()) / (np.ptp(ys_np) + 1e-8)

        # 3) area check
        area = 0.5 * abs(np.dot(xs_u, np.roll(ys_u, 1)) - np.dot(ys_u, np.roll(xs_u, 1)))
        if area < K:
            continue

        # 4) quadrant scaling
        cx, cy = 0.5, 0.5
        s = {q: random.uniform(min_scale, 1.0) for q in ['tr', 'tl', 'bl', 'br']}
        xs_s, ys_s = [], []
        for x0, y0 in zip(xs_u, ys_u):
            dx, dy = x0 - cx, y0 - cy
            if dx >= 0 and dy >= 0:
                scale = s['tr']
            elif dx < 0 and dy >= 0:
                scale = s['tl']
            elif dx < 0 and dy < 0:
                scale = s['bl']
            else:
                scale = s['br']
            xs_s.append(cx + dx * scale)
            ys_s.append(cy + dy * scale)
        xs_s = np.array(xs_s)
        ys_s = np.array(ys_s)

        # 5) scale to box
        xs_f = xs_s * width
        ys_f = ys_s * height

        # 6) random placement
        minx, miny = xs_f.min(), ys_f.min()
        w_box, h_box = xs_f.max() - minx, ys_f.max() - miny
        opts = [
            (0, 0),
            (width - w_box, 0),
            (0, height - h_box),
            (width - w_box, height - h_box),
            ((width - w_box) / 2, (height - h_box) / 2)
        ]
        ox, oy = random.choice(opts)
        xs_final = xs_f - minx + ox
        ys_final = ys_f - miny + oy
        return xs_final, ys_final

    # fallback last shape
    return xs_final, ys_final


def snap_to_corner(
    xs: np.ndarray,
    ys: np.ndarray,
    corner: str,
    width: float,
    height: float,
    thr: float = 0.025
) -> tuple[np.ndarray, np.ndarray]:
    """
    Snap the closest polygon vertex to the given corner, and its two
    immediate neighbors along the hull to their respective borders.
    If a neighbor is already clamped (i.e. exactly on any cell border), skip it.
    """
    n = len(xs)
    xs = xs.copy()
    ys = ys.copy()

    # Determine absolute threshold distances
    abs_x_thr = thr
    abs_y_thr = thr

    # Map corner to target coords and neighbor offsets
    if corner == 'bottom-left':
        tx, ty = 0.0, 0.0
        x_dirs = [0, -1, -2]
        y_dirs = [0, +1, +2]
        x_val, y_val = 0.0, 0.0
    elif corner == 'bottom-right':
        tx, ty = width, 0.0
        x_dirs = [0, +1, +2]
        y_dirs = [0, -1, -2]
        x_val, y_val = width, 0.0
    elif corner == 'top-left':
        tx, ty = 0.0, height
        x_dirs = [0, +1, +2]
        y_dirs = [0, -1, -2]
        x_val, y_val = 0.0, height
    elif corner == 'top-right':
        tx, ty = width, height
        x_dirs = [0, -1, -2]
        y_dirs = [0, +1, +2]
        x_val, y_val = width, height
    else:
        raise ValueError(f"Unknown corner: {corner}")

    # Find the index of the closest vertex
    dists = np.hypot(xs - tx, ys - ty)
    i0 = int(np.argmin(dists))

    # Always clamp the central vertex to the corner
    xs[i0], ys[i0] = x_val, y_val

    # Helper to detect if a vertex is already on any border
    def is_on_border(x, y):
        return (np.isclose(x, 0.0) or np.isclose(x, width)
                or np.isclose(y, 0.0) or np.isclose(y, height))

    # Clamp X for horizontal neighbors, but only if not already on a border
    for dx in x_dirs[1:]:
        idx = (i0 + dx) % n
        if not is_on_border(xs[idx], ys[idx]):
            xs[idx] = x_val

    # Clamp Y for vertical neighbors, but only if not already on a border
    for dy in y_dirs[1:]:
        idx = (i0 + dy) % n
        if not is_on_border(xs[idx], ys[idx]):
            ys[idx] = y_val

    return xs, ys


def snap_to_border(
    xs: np.ndarray,
    ys: np.ndarray,
    border: str,
    width: float,
    height: float,
    thr: float = 0.05
) -> tuple[np.ndarray, np.ndarray]:
    """
    Snap polygon vertices to the specified border.
    `border` ∈ {'bottom','top','left','right'}.
    `thr` is treated as a fraction of cell height (for top/bottom)
    or cell width (for left/right). Only vertices within that
    threshold get clamped.
    """
    xs_b, ys_b = xs.copy(), ys.copy()

    if border in ('bottom', 'top'):
        abs_thr = thr * height
        if border == 'bottom':
            ys_b[ys_b < abs_thr] = 0
        else:  # 'top'
            ys_b[ys_b > height - abs_thr] = height

    elif border in ('left', 'right'):
        abs_thr = thr * width
        if border == 'left':
            xs_b[xs_b < abs_thr] = 0
        else:  # 'right'
            xs_b[xs_b > width - abs_thr] = width

    else:
        raise ValueError(f"Unknown border: {border}")

    return xs_b, ys_b
