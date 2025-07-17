import random
import matplotlib.pyplot as plt

def random_partition(total, count, min_frac=0.0, max_frac=1.0, max_attempts=100000):
    """
    Partition `total` into `count` segments, each between:
      - min_frac * total  (inclusive)
      - max_frac * total  (inclusive)
    The segments sum exactly to `total`.

    Raises ValueError if constraints cannot be satisfied.
    """
    # Validate fraction constraints
    if not (0 <= min_frac <= max_frac <= 1):
        raise ValueError(f"min_frac and max_frac must satisfy 0 <= min_frac <= max_frac <= 1, got {min_frac}, {max_frac}")
    if count * min_frac > 1:
        raise ValueError(f"Cannot partition {total} into {count} parts: min_frac {min_frac} too large.")
    if count * max_frac < 1:
        raise ValueError(f"Cannot partition {total} into {count} parts: max_frac {max_frac} too small.")

    for _ in range(max_attempts):
        cuts = sorted([0.0] + [random.random() for _ in range(count - 1)] + [1.0])
        parts = [(cuts[i+1] - cuts[i]) * total for i in range(count)]
        if all(min_frac * total <= p <= max_frac * total for p in parts):
            return parts

    raise ValueError(
        f"Failed to partition {total} into {count} parts "
        f"with min_frac {min_frac} and max_frac {max_frac} after {max_attempts} attempts"
    )

def divide_cross_section(
    X=0.7,
    Y=0.5,
    n_rows=2,
    min_div=2,
    max_div=3,
    min_width_frac=0.2,
    min_height_frac=0.2,
    TS_position='nrandom',
    max_partition_attempts=100000,
):
    """
    Divide a rectangle of width X and height Y into horizontal stripes,
    each stripe randomly partitioned into columns. Optionally include a
    'Through Stone' (TS) column spanning the full height, using the
    top stripe to fix TS width for all stripes.

    Parameters:
      X, Y: overall dimensions
      n_rows: number of horizontal stripes
      min_div, max_div: min/max subdivisions per stripe (excluding TS)
      min_width_frac: min fraction of X for any column (including TS)
      min_height_frac: min fraction of Y for each stripe
      TS_position: 'none', 'nrandom' (random including none), 'random', 'left', 'center', or 'right'
      max_partition_attempts: max attempts for random partitions

    Returns:
      Xlimits_rows: list of lists of column widths per stripe
      Ylimits_rows: list of lists of column heights per stripe
    """
    print()
    valid_positions = ('none','nrandom', 'random','left','center','right')
    pos = TS_position.lower()
    print(pos)
    if pos not in valid_positions:
        raise ValueError(f"TS_position must be one of {valid_positions}, got {TS_position}")
    if pos == 'random':
        if max_div < 3:
            pos = random.choice(['left','right'])
        else:
            pos = random.choice(['left','center', 'center', 'center', 'right'])
    elif pos == 'nrandom':
        if max_div < 3:
            pos = random.choice(['left','right', 'none'])
        else:
            pos = random.choice(['left','center', 'center', 'center', 'right', 'none', 'none'])

    # Validate fractions and counts
    if not (0 <= min_height_frac <= 1):
        raise ValueError("min_height_frac must be between 0 and 1")
    if not (0 <= min_width_frac <= 1):
        raise ValueError("min_width_frac must be between 0 and 1")
    if n_rows * min_height_frac > 1:
        raise ValueError(
            f"Cannot split into {n_rows} rows with minimum height fraction {min_height_frac}"
        )
    if min_div < 1 or max_div < min_div:
        raise ValueError("Invalid column division range")
    max_TS_frac = 1.0 - (max_div - 1) * (min_width_frac + 0.01)
    if min_width_frac > max_TS_frac:
        raise ValueError(
            f"min_width_frac {min_width_frac} too large for TS position {pos} "
            f"with max_div {max_div} (with margin 0.01)"
        )
    print(f"max width fraction for TS: {max_TS_frac}")

    # Split height evenly with random constraints
    stripe_heights = random_partition(Y, n_rows, min_height_frac, 1.0, max_partition_attempts)
    Xlimits_rows = []
    Ylimits_rows = []

    # No TS case: independent random partitions per stripe
    if pos == 'none':
        for h in stripe_heights:
            n_cols = random.randint(min_div, max_div)
            widths = random_partition(X, n_cols, min_width_frac, 1.0, max_partition_attempts)
            Xlimits_rows.append(widths)
            Ylimits_rows.append([h]*n_cols)
        return Xlimits_rows, Ylimits_rows

    # For TS cases: use top stripe to fix TS width
    # --------------------- LEFT or RIGHT TS ---------------------
    if pos in ('left','right'):
        # decide subdivisions excluding TS on top row
        total_cols_top = random.randint(min_div, max_div)
        top_widths = random_partition(X, total_cols_top, min_width_frac, max_TS_frac, max_partition_attempts)
        # extract TS width from first or last
        if pos == 'left':
            ts_width = top_widths[0]
            Xlimits_rows.append(top_widths)
        else:
            ts_width = top_widths[-1]
            Xlimits_rows.append(top_widths)
        Ylimits_rows.append([stripe_heights[0]]*total_cols_top)

        print(f"TS width: {ts_width}, position: {pos}")
        print(f"Remeining width: {X - ts_width}")
        rem_width = X - ts_width
        # remaining stripes
        for h in stripe_heights[1:]:
            sub = random.randint(max(min_div-1, 1), max(max_div-1, 1))
            rem_parts = random_partition(rem_width, sub, min_width_frac * X / rem_width, 1.0, max_partition_attempts)
            if pos == 'left':
                widths = [ts_width] + rem_parts
            else:
                widths = rem_parts + [ts_width]
            Xlimits_rows.append(widths)
            Ylimits_rows.append([h]*len(widths))
        return Xlimits_rows, Ylimits_rows

     # --------------------- CENTER TS ---------------------
    if max_div < 3:
        raise ValueError("For center TS, max_div must be at least 3 to allow for left and right subdivisions")
    # ensure top stripe has room for two sides + TS
    min_top = max(3, min_div)
    total_cols_top = random.randint(min_top, max_div)
    left_top  = random.randint(1, total_cols_top - 2)
    right_top = total_cols_top - left_top - 1
    top_widths = random_partition(X, total_cols_top, min_width_frac, max_TS_frac, max_partition_attempts)
    
    ts_index = left_top
    ts_width  = top_widths[ts_index]
    left_rem  = sum(top_widths[:ts_index])
    right_rem = sum(top_widths[ts_index+1:])
    
    Xlimits_rows.append(top_widths)
    Ylimits_rows.append([stripe_heights[0]] * total_cols_top)

    # remaining stripes
    for h in stripe_heights[1:]:
        # compute side‐region max counts based on width & min frac
        max_left_sub  = min(max_div - 2, int(left_rem  // (min_width_frac * X)))
        max_right_sub = min(max_div - 2, int(right_rem // (min_width_frac * X)))

        # build all valid (l, r) so total columns stays within [min_div, max_div]
        valid_pairs = [
            (l, r)
            for l in range(1, max_left_sub+1)
            for r in range(1, max_right_sub+1)
            if min_div - 1 <= l + r <= max_div - 1
        ]
        if not valid_pairs:
            raise ValueError(
                f"No valid subdivisions on stripe height {h:.3f}: "
                f"left_rem={left_rem:.3f}, right_rem={right_rem:.3f}, "
                f"max_left_sub={max_left_sub}, max_right_sub={max_right_sub}"
            )

        left_sub, right_sub = random.choice(valid_pairs)

        # partition each side‐region
        left_parts  = random_partition(
            left_rem,  left_sub,
            min_width_frac * X / left_rem, 1.0, max_partition_attempts
        )
        right_parts = random_partition(
            right_rem, right_sub,
            min_width_frac * X / right_rem, 1.0, max_partition_attempts
        )

        widths = left_parts + [ts_width] + right_parts
        Xlimits_rows.append(widths)
        Ylimits_rows.append([h] * len(widths))

    return Xlimits_rows, Ylimits_rows

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Parameters
    X, Y = 0.06, 0.04
    n_rows = 2
    min_div, max_div = 2, 4
    min_width_frac = 0.1    # no cell narrower than 20% of X
    min_height_frac = 0.3    # no stripe shorter than 20% of Y
    TS_position = 'random'     # fixed TS position for all

    # Create a 2x5 grid of subplots for 10 images
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()

    for ax in axes:
        # Generate one subdivision
        Xlims, Ylims = divide_cross_section(
            X, Y, n_rows,
            min_div, max_div,
            min_width_frac,
            min_height_frac,
            TS_position=TS_position,
        )
        # Compute Y-offsets for stripes
        y_off = [0]
        for stripe_heights in Ylims[:-1]:
            y_off.append(y_off[-1] + stripe_heights[0])

        # Draw rectangles
        for row_idx, (widths, heights) in enumerate(zip(Xlims, Ylims)):
            x0 = 0
            for w, h in zip(widths, heights):
                ax.add_patch(plt.Rectangle((x0, y_off[row_idx]), w, h,
                                        fill=False, edgecolor='black'))
                x0 += w

        ax.set_xlim(0, X)
        ax.set_ylim(0, Y)
        ax.set_aspect('equal')
        ax.axis('off')

    plt.tight_layout()
    plt.show()