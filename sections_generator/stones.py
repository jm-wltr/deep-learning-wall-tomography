import random
import numpy as np


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


def _build_edge_vectors(chain: list[int]) -> list[tuple[int, int]]:
    """
    Turn a chain of coords into a list of adjacent edge pairs.
    """
    return [(chain[i], chain[i+1]) for i in range(len(chain)-1)]


def first_non_consecutive(indices: list[int]) -> tuple[int, int]:
    """
    Return (position, value) of the first non-consecutive element in `indices`.
    If all are consecutive, returns (0, 1).
    """
    for idx in range(1, len(indices)):
        if indices[idx] - indices[idx-1] != 1:
            return idx, indices[idx]
    return 0, 1


def random_stone(
    sides: int = 12,
    width: float = 0.35,
    height: float = 0.25,
    K: float = 0.75
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a random non-self-intersecting polygon (a 'stone') with `sides` edges,
    fitting in a box of size (width, height), and ensure it covers at least `K`
    fraction of the cell area.
    """
    for _ in range(100):  # retry up to 100 times
        # 1) sample and split coords
        x_coords = _sample_sorted_coords(sides)
        y_coords = _sample_sorted_coords(sides)
        xa, xb = _split_chain(x_coords)
        ya, yb = _split_chain(y_coords)

        xb = xb[::-1]
        yb = yb[::-1]

        x_vectors = _build_edge_vectors(xa) + _build_edge_vectors(xb)
        y_vectors = _build_edge_vectors(ya) + _build_edge_vectors(yb)
        random.shuffle(y_vectors)

        edges = []
        for (x0, x1), (y0, y1) in zip(x_vectors, y_vectors):
            dx = x1 - x0
            dy = y1 - y0
            edges.append((np.arctan2(dy, dx), np.hypot(dx, dy)))
        edges.sort(key=lambda e: e[0])

        angles, lengths = zip(*edges)
        xs = [0.0]; ys = [0.0]
        for ang, ln in zip(angles, lengths):
            xs.append(xs[-1] + ln * np.cos(ang))
            ys.append(ys[-1] + ln * np.sin(ang))

        xs_arr = np.array(xs[:-1])
        ys_arr = np.array(ys[:-1])

        # 2) Normalize to box
        xs_arr = (xs_arr - xs_arr.min()) / (xs_arr.max() - xs_arr.min() + 1e-8)
        ys_arr = (ys_arr - ys_arr.min()) / (ys_arr.max() - ys_arr.min() + 1e-8)

        # 3) Compute area
        area_unit = 0.5 * np.abs(np.dot(xs_arr, np.roll(ys_arr, 1)) - np.dot(ys_arr, np.roll(xs_arr, 1)))

        if area_unit >= K:
            # Scale to width × height
            return xs_arr * width, ys_arr * height

    # If all attempts fail, return scaled-down shape
    return xs_arr * width, ys_arr * height



def snap_to_corner(
    xs: np.ndarray,
    ys: np.ndarray,
    corner: str,
    width: float,
    height: float,
    thr: float = 0.025
) -> tuple[np.ndarray, np.ndarray]:
    """
    Snap polygon vertices to the specified corner of the bounding box.
    Safe wrap-around indexing. Returns `sides` points, no duplicate.
    Parameters:
      thr: threshold distance for preliminary border snapping
    """
    n = len(xs)
    xs = xs.copy()
    ys = ys.copy()

    # preliminary border snaps
    if 'bottom' in corner:
        ys[ys < thr] = 0
    if 'top' in corner:
        ys[ys > height - thr] = height
    if 'left' in corner:
        xs[xs < thr] = 0
    if 'right' in corner:
        xs[xs > width - thr] = width

    # pick target corner
    if corner == 'bottom-left':
        target = (0.0, 0.0)
        x_offs, y_offs, x_val, y_val = [0, -1, -2], [0, +1, +2], 0, 0
    elif corner == 'bottom-right':
        target = (width, 0.0)
        x_offs, y_offs, x_val, y_val = [0, +1, +2], [0, -1, -2], width, 0
    elif corner == 'top-left':
        target = (0.0, height)
        x_offs, y_offs, x_val, y_val = [0, +1, +2], [0, -1, -2], 0, height
    elif corner == 'top-right':
        target = (width, height)
        x_offs, y_offs, x_val, y_val = [0, -1, -2], [0, +1, +2], width, height
    else:
        raise ValueError(f"Unknown corner: {corner}")

    # find closest vertex
    dists = np.hypot(xs - target[0], ys - target[1])
    i0 = int(np.argmin(dists))

    # wrap indices and assign
    xi = [(i0 + dx) % n for dx in x_offs]
    yi = [(i0 + dy) % n for dy in y_offs]
    xs[xi] = x_val
    ys[yi] = y_val

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
    Snap polygon vertices to the specified horizontal border.
    Safe wrap-around indexing. Returns `sides` points, no duplicate.
    Parameters:
      thr: threshold distance for border proximity snapping
    """
    n = len(xs)
    xs = xs.copy()
    ys = ys.copy()

    if border == 'bottom':
        on = np.where(ys < thr)[0]
        ys[on] = 0
        if on.size > 0:
            i_min, i_max = on.min(), on.max()
            if xs[i_min] - xs[(i_min-1)%n] > thr:
                ys[(i_min-1)%n] = 0
            if xs[(i_max+1)%n] - xs[i_max] > thr:
                ys[(i_max+1)%n] = 0
    elif border == 'top':
        on = np.where(ys > height - thr)[0]
        ys[on] = height
        if on.size == 0:
            ys[0] = height
            ys[-1] = height
        elif on.size == 1:
            ys[0] = height
            if xs[-1] - xs[0] > thr:
                ys[-1] = height
        elif on.size == 2:
            ys[[0, 1]] = height
            if xs[-1] - xs[0] > thr:
                ys[-1] = height
        else:
            m, ncn = first_non_consecutive(on.tolist())
            if xs[ncn-1] - xs[ncn] > thr:
                ys[(ncn-1)%n] = height
            if xs[m-1] - xs[m] > thr:
                ys[m % n] = height
    else:
        raise ValueError(f"Unknown border: {border}")

    return xs, ys

def clip_polygon_to_rect(
    xs: np.ndarray,
    ys: np.ndarray,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    eps: float = 1e-5,
    min_dist: float = 1e-9
) -> tuple[np.ndarray, np.ndarray]:
    """
    Clip an open polygon defined by (xs, ys) to the rectangle
    [x_min,x_max] x [y_min,y_max] using the Sutherland-Hodgman algorithm.

    After clipping, clamps vertices within a small epsilon to avoid exact overlaps.
    If any pair of consecutive points is too close (under `min_dist`), raises ValueError.
    Returns clipped xs, ys arrays (may have different length).
    """

    def _clip_edge(points, edge_fn, intersect_fn):
        out = []
        if not points:
            return out
        prev = points[-1]
        prev_inside = edge_fn(prev)
        for curr in points:
            curr_inside = edge_fn(curr)
            if curr_inside:
                if not prev_inside:
                    out.append(intersect_fn(prev, curr))
                out.append(curr)
            elif prev_inside:
                out.append(intersect_fn(prev, curr))
            prev, prev_inside = curr, curr_inside
        return out

    pts = list(zip(xs, ys))

    # Perform Sutherland-Hodgman clipping
    pts = _clip_edge(pts, lambda p: p[0] >= x_min, lambda p, q: (x_min, p[1] + (q[1]-p[1])*(x_min-p[0])/(q[0]-p[0])))
    pts = _clip_edge(pts, lambda p: p[0] <= x_max, lambda p, q: (x_max, p[1] + (q[1]-p[1])*(x_max-p[0])/(q[0]-p[0])))
    pts = _clip_edge(pts, lambda p: p[1] >= y_min, lambda p, q: (p[0] + (q[0]-p[0])*(y_min-p[1])/(q[1]-p[1]), y_min))
    pts = _clip_edge(pts, lambda p: p[1] <= y_max, lambda p, q: (p[0] + (q[0]-p[0])*(y_max-p[1])/(q[1]-p[1]), y_max))

    if not pts:
        return np.empty(0), np.empty(0)

    # Clamp slightly inward
    clamped = []
    for x, y in pts:
        x = min(max(x, x_min + eps), x_max - eps)
        y = min(max(y, y_min + eps), y_max - eps)
        clamped.append((x, y))

    # Remove near-duplicate consecutive points
    filtered = [clamped[0]]
    for pt in clamped[1:]:
        if abs(pt[0] - filtered[-1][0]) > eps or abs(pt[1] - filtered[-1][1]) > eps:
            filtered.append(pt)

    # If closed polygon (first == last), remove duplicate endpoint
    if len(filtered) > 1 and np.allclose(filtered[0], filtered[-1], atol=eps):
        filtered.pop()

    # Final check for degenerate short edges
    for i in range(len(filtered)):
        p1 = filtered[i]
        p2 = filtered[(i + 1) % len(filtered)]
        dist = np.hypot(p2[0] - p1[0], p2[1] - p1[1])
        if dist < min_dist:
            raise ValueError(f"Degenerate peak detected: edge {i}-{(i+1)%len(filtered)} is too short ({dist:.6f})")

    xs_out, ys_out = zip(*filtered)
    return np.array(xs_out), np.array(ys_out)
