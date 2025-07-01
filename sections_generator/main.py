import os
import ezdxf
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon, Rectangle
from section import generate_cross_section

# Color mappings (RGB)
colors = {
    "Piedra": [172, 96, 73],   # stone color
    "Mortero": [37, 173, 221]  # mortar/background color
}
# Normalize to 0-1 range
stone_color = tuple(c/255 for c in colors["Piedra"])
mortar_color = tuple(c/255 for c in colors["Mortero"])


def export_to_dxf(stones, path, filename):
    """Export list of stone polygons to a DXF file."""
    os.makedirs(path, exist_ok=True)
    doc = ezdxf.new("R2010", setup=True)
    msp = doc.modelspace()
    for coords in stones:
        for i in range(len(coords)):
            start = coords[i]
            end = coords[(i + 1) % len(coords)]
            msp.add_line(start, end)
    doc.saveas(os.path.join(path, filename + '.dxf'))


def export_to_jpg(stones, X, Y, path, filename):
    """Export stone polygons to a JPG image with stones filled stone_color and background mortar_color."""
    os.makedirs(path, exist_ok=True)
    fig, ax = plt.subplots(figsize=(X * 4, Y * 4))

    # Draw mortar background
    ax.add_patch(Rectangle((0, 0), X, Y, facecolor=mortar_color, edgecolor=None))

    # Draw stones filled with stone_color, no edge
    for coords in stones:
        poly = MplPolygon(coords, closed=True, facecolor=stone_color, edgecolor=None)
        ax.add_patch(poly)

    ax.set_xlim(0, X)
    ax.set_ylim(0, Y)
    ax.set_aspect('equal')
    ax.axis('off')

    outpath = os.path.join(path, filename + '.jpg')
    plt.tight_layout(pad=0)
    fig.savefig(outpath, dpi=600)
    plt.close(fig)


def main():
    # Parameters
    X = 1.4
    Y = 0.3
    K = 0.1  # minimum stone area fraction
    sides = 15  # sides per stone polygon
    n = 20  # number of cross-sections to generate
    n_rows = 2
    min_div = 4
    max_div = 7
    min_width_frac = 0.05
    min_height_frac = 0.2
    max_partition_attempts = 10000
    TS_mode = True  # whether to include some TS sections or not

    # Output directories
    base_dir = 'sections_generator/output'
    dxf_dir = os.path.join(base_dir, 'sections')
    jpg_dir = os.path.join(base_dir, 'images')

    count = 0
    attempts = 0
    max_total_attempts = n * 10  # safeguard to avoid infinite loops

    while count < n and attempts < max_total_attempts:
        attempts += 1
        TS_position = 'center' if TS_mode else 'none'

        try:
            _, _, _, stones = generate_cross_section(
                X=X, Y=Y,
                n_rows=n_rows,
                min_div=min_div,
                max_div=max_div,
                min_width_frac=min_width_frac,
                min_height_frac=min_height_frac,
                TS_position=TS_position,
                max_partition_attempts=max_partition_attempts,
                sides=sides,
                K=K
            )
        except ValueError as e:
            print(f"[{count}] Skipping due to ValueError: {e}")
            continue

        fname = f'{count:05d}'
        export_to_dxf(stones, dxf_dir, fname)
        export_to_jpg(stones, X, Y, jpg_dir, fname)
        count += 1

    print(f"Saved {count} sections: DXF in '{dxf_dir}', JPG in '{jpg_dir}'")


if __name__ == '__main__':
    main()
