import os
import ezdxf
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon, Rectangle
from PIL import Image
from section import generate_cross_section  # or your actual module

# Color mappings (RGB)
colors = {
    "Piedra": [75, 99, 171],   # stone color
    "Mortero": [219, 171, 36]  # mortar/background color
}
stone_color  = tuple(c/255 for c in colors["Piedra"])
mortar_color = tuple(c/255 for c in colors["Mortero"])

# Configuration for each generation mode
MODE_CONFIGS = {
    "normal": {
        # Section size (X, Y)
        "size": (0.7, 0.5),
        # generate_cross_section arguments
        "n_rows": 2,
        "min_div": 2,
        "max_div": 3,
        "min_width_frac": 0.2,
        "min_height_frac": 0.2,
        "TS_position": "center",
        "max_partition_attempts": 10000,
        "sides": 15,
        "K": 0.5,
        # output directories
        "dxf_subdir": "sectionsNormal",
        "jpg_subdir": "imagesNormal",
    },
    "crop": {
        # Full section dimensions
        "full_size": (1.4, 0.5),
        # Crop window dimensions and step
        "crop_size": (0.7, 0.5),
        "crop_step": 0.1,
        # generate_cross_section arguments
        "n_rows": 2,
        "min_div": 4,
        "max_div": 8,
        "min_width_frac": 0.05,
        "min_height_frac": 0.2,
        "TS_position": "center",
        "max_partition_attempts": 10000,
        "sides": 15,
        "K": 0.5,
        # output directories
        "dxf_subdir": "sectionsCrop",
        "jpg_subdir": "imagesCrop",
    },
    "rotate": {
        # Rotated section flips dimensions
        "size": (0.5, 0.7),  # swapped from normal.crop_size
        # generate_cross_section arguments
        "n_rows": 4,
        "min_div": 2,
        "max_div": 2,
        "min_width_frac": 0.2,
        "min_height_frac": 0.1,
        "TS_position": "none",
        "max_partition_attempts": 10000,
        "sides": 15,
        "K": 0.5,
        # output directories
        "dxf_subdir": "sectionsRot",
        "jpg_subdir": "imagesRotate",
    }
}

BASE_DIR = "sections_generator/output"

def export_to_dxf(stones, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    doc = ezdxf.new("R2010", setup=True)
    msp = doc.modelspace()
    for poly in stones:
        for i in range(len(poly)):
            a = poly[i]
            b = poly[(i+1) % len(poly)]
            msp.add_line(a, b)
    filepath = os.path.join(out_dir, f"{name}.dxf")
    doc.saveas(filepath)
    return filepath


def export_to_jpg(stones, size, out_dir, name, xlim=None, ylim=None):
    os.makedirs(out_dir, exist_ok=True)
    width, height = size
    fig, ax = plt.subplots(figsize=(width*4, height*4))

    # Determine background area
    if xlim and ylim:
        x0, x1 = xlim
        y0, y1 = ylim
        bg_origin = (x0, y0)
        bg_width = x1 - x0
        bg_height = y1 - y0
    else:
        bg_origin = (0, 0)
        bg_width, bg_height = width, height

    ax.add_patch(Rectangle(bg_origin, bg_width, bg_height,
                           facecolor=mortar_color, edgecolor=None))

    for poly in stones:
        ax.add_patch(MplPolygon(poly, closed=True,
                                 facecolor=stone_color, edgecolor=None))

    ax.set_xlim(*(xlim or (0, width)))
    ax.set_ylim(*(ylim or (0, height)))
    ax.set_aspect("equal")
    ax.axis("off")

    out_path = os.path.join(out_dir, f"{name}.jpg")
    plt.tight_layout(pad=0)
    fig.savefig(out_path, dpi=600)
    plt.close(fig)
    return out_path


def process_normal(cfg, count):
    size = cfg["size"]
    stones = generate_cross_section(
        X=size[0], Y=size[1],
        n_rows=cfg["n_rows"],
        min_div=cfg["min_div"], max_div=cfg["max_div"],
        min_width_frac=cfg["min_width_frac"],
        min_height_frac=cfg["min_height_frac"],
        TS_position=cfg["TS_position"],
        max_partition_attempts=cfg["max_partition_attempts"],
        sides=cfg["sides"],
        K=cfg["K"]
    )[-1]
    name = f"{count:05d}"
    dxf_path = export_to_dxf(stones,
                              os.path.join(BASE_DIR, cfg["dxf_subdir"]),
                              name)
    jpg_path = export_to_jpg(stones, size,
                              os.path.join(BASE_DIR, cfg["jpg_subdir"]),
                              name)


def process_crop(cfg, count):
    full_size = cfg["full_size"]
    stones_full = generate_cross_section(
        X=full_size[0], Y=full_size[1],
        n_rows=cfg["n_rows"],
        min_div=cfg["min_div"], max_div=cfg["max_div"],
        min_width_frac=cfg["min_width_frac"],
        min_height_frac=cfg["min_height_frac"],
        TS_position=cfg["TS_position"],
        max_partition_attempts=cfg["max_partition_attempts"],
        sides=cfg["sides"],
        K=cfg["K"]
    )[-1]
    base_name = f"{count:05d}_full"
    export_to_dxf(stones_full,
                  os.path.join(BASE_DIR, cfg["dxf_subdir"]),
                  base_name)

    wx, hy = cfg["crop_size"]
    x = 0.0
    idx = 0
    while x <= full_size[0] - wx + 1e-6:
        name = f"{count:05d}_crop{idx}"
        export_to_jpg(stones_full, cfg["crop_size"],
                      os.path.join(BASE_DIR, cfg["jpg_subdir"]),
                      name,
                      xlim=(x, x+wx),
                      ylim=(0, hy))
        x += cfg["crop_step"]
        idx += 1


def process_rotate(cfg, count):
    size = cfg["size"]
    stones_r = generate_cross_section(
        X=size[0], Y=size[1],
        n_rows=random.randint(2, 4),  # Random number of rows
        min_div=cfg["min_div"], max_div=cfg["max_div"],
        min_width_frac=cfg["min_width_frac"],
        min_height_frac=cfg["min_height_frac"],
        TS_position=cfg["TS_position"],
        max_partition_attempts=cfg["max_partition_attempts"],
        sides=cfg["sides"],
        K=cfg["K"]
    )[-1]
    name = f"{count:05d}_rot"
    dxf_path = export_to_dxf(stones_r,
                              os.path.join(BASE_DIR, cfg["dxf_subdir"]),
                              name)
    jpg_path = export_to_jpg(stones_r, size,
                              os.path.join(BASE_DIR, cfg["jpg_subdir"]),
                              name)
    # Additionally save a 90° CW rotated image
    img = Image.open(jpg_path)
    img.rotate(-90, expand=True).save(jpg_path)


def main():
    # Choose mode: 'normal', 'crop', or 'rotate'
    generation_mode = "rotate"  # <-- modify as needed

    cfg = MODE_CONFIGS.get(generation_mode)
    if cfg is None:
        raise ValueError(f"Unknown mode '{generation_mode}'")

    target_count = 10  # number of sections to generate
    count = 0
    attempts = 0
    max_attempts = target_count * 20

    while count < target_count and attempts < max_attempts:
        attempts += 1
        try:
            if generation_mode == "normal":
                process_normal(cfg, count)
            elif generation_mode == "crop":
                process_crop(cfg, count)
            elif generation_mode == "rotate":
                process_rotate(cfg, count)

            count += 1
        except ValueError as e:
            print(f"[{count}] skipped: {e}")
            continue

    print(f"Generated {count}/{target_count} sections in mode={generation_mode}")

if __name__ == "__main__":
    main()