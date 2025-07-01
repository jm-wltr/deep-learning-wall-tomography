import os
import ezdxf
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon, Rectangle
from PIL import Image
from shapely.geometry import Polygon as ShapelyPolygon, box
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
        "size": (0.6, 0.4),
        "n_rows": 2,
        "min_div": 2,
        "max_div": 4,
        "min_width_frac": 0.2,
        "min_height_frac": 0.2,
        "TS_position": "center",
        "max_partition_attempts": 10000,
        "sides": 15,
        "K": 0.5,
        "dxf_subdir": "sectionsNormal",
        "jpg_subdir": "imagesNormal",
    },
    "crop": {
        "full_size": (1.2, 0.4),
        "crop_size": (0.6, 0.4),
        "crop_step": 0.1,
        "n_rows": 2,
        "min_div": 4,
        "max_div": 7,
        "min_width_frac": 0.1,
        "min_height_frac": 0.2,
        "TS_position": "center",
        "max_partition_attempts": 10000,
        "sides": 15,
        "K": 0.5,
        "dxf_subdir": "sectionsCrop",
        "jpg_subdir": "imagesCrop",
    },
    "rotate": {
        "size": (0.4, 0.6),
        "n_rows": 4,
        "min_div": 2,
        "max_div": 2,
        "min_width_frac": 0.2,
        "min_height_frac": 0.2,
        "TS_position": "none",
        "max_partition_attempts": 10000,
        "sides": 15,
        "K": 0.5,
        "dxf_subdir": "sectionsRot",
        "jpg_subdir": "imagesRotate",
    }
}

BASE_DIR = "sections_generator/output"

def export_to_dxf(stones, size, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    doc = ezdxf.new("R2010", setup=True)
    msp = doc.modelspace()

    # Draw bounding box
    x_min, y_min = 0.0, 0.0
    x_max, y_max = size
    bbox = [(x_min, y_min), (x_min, y_max), (x_max, y_max), (x_max, y_min), (x_min, y_min)]
    msp.add_lwpolyline(bbox, dxfattribs={"layer": "BOUNDING_BOX"})

    # Draw stones
    for poly in stones:
        if len(poly) < 2:
            continue
        for i in range(len(poly)):
            p1 = poly[i]
            p2 = poly[(i+1) % len(poly)]
            msp.add_line(p1, p2, dxfattribs={"layer": "STONE"})

    path = os.path.join(out_dir, f"{name}.dxf")
    doc.saveas(path)
    return path


def export_to_jpg(stones, size, out_dir, name, xlim=None, ylim=None):
    os.makedirs(out_dir, exist_ok=True)
    width, height = size
    fig, ax = plt.subplots(figsize=(width*4, height*4))

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

    out = os.path.join(out_dir, f"{name}.jpg")
    plt.tight_layout(pad=0)
    fig.savefig(out, dpi=600)
    plt.close(fig)
    return out


def process_crop(cfg, count):
    full_size = cfg["full_size"]
    crop_w, crop_h = cfg["crop_size"]
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

    # Optional: export full DXF
    base_name = f"{count:05d}_full"
    export_to_dxf(stones_full, full_size,
                  os.path.join(BASE_DIR, cfg["dxf_subdir"]),
                  base_name)

    x = 0.0
    idx = 0
    while x <= full_size[0] - crop_w + 1e-6:
        # Define crop window
        crop_box = box(x, 0.0, x + crop_w, crop_h)
        # Clip each stone polygon
        clipped = []
        for poly in stones_full:
            shp = ShapelyPolygon(poly)
            inter = shp.intersection(crop_box)
            if not inter.is_empty:
                if hasattr(inter, 'geoms'):
                    for part in inter.geoms:
                        clipped.append(list(part.exterior.coords))
                else:
                    clipped.append(list(inter.exterior.coords))

        # Export cropped DXF and JPG
        name = f"{count:05d}_crop{idx}"
        export_to_dxf(clipped, (crop_w, crop_h),
                      os.path.join(BASE_DIR, cfg["dxf_subdir"]), name)
        export_to_jpg(clipped, (crop_w, crop_h),
                      os.path.join(BASE_DIR, cfg["jpg_subdir"]), name,
                      xlim=(x, x+crop_w), ylim=(0, crop_h))

        x += cfg["crop_step"]
        idx += 1

    return


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
    dxf_path = export_to_dxf(stones, (size[0], size[1]),
                              os.path.join(BASE_DIR, cfg["dxf_subdir"]),
                              name)
    jpg_path = export_to_jpg(stones, size,
                              os.path.join(BASE_DIR, cfg["jpg_subdir"]),
                              name)



def process_rotate(cfg, count):
    size = cfg["size"]  # (width, height)
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
    # Export original DXF if needed (optional)
    # export_to_dxf(stones_r, size,
    #               os.path.join(BASE_DIR, cfg["dxf_subdir"]), name)

    # Rotate stone coordinates 90° CW around origin and swap bbox dims
    w, h = size
    rotated = []
    for poly in stones_r:
        rotated_poly = [(y, w - x) for x, y in poly]
        rotated.append(rotated_poly)
    # New size after rotation
    new_size = (h, w)
    out_dir = os.path.join(BASE_DIR, cfg["dxf_subdir"])
    # Export rotated DXF
    export_to_dxf(rotated, new_size, out_dir, name)

    # Export JPG by rotating the saved image for consistency
    jpg_dir = os.path.join(BASE_DIR, cfg["jpg_subdir"])
    export_to_jpg(stones_r, size, jpg_dir, name)
    img = Image.open(os.path.join(jpg_dir, f"{name}.jpg"))
    img.rotate(-90, expand=True).save(os.path.join(jpg_dir, f"{name}.jpg"))




def main():
    # Choose mode: 'normal', 'crop', or 'rotate'
    generation_mode = "rotate"  # <-- modify as needed

    cfg = MODE_CONFIGS.get(generation_mode)
    if cfg is None:
        raise ValueError(f"Unknown mode '{generation_mode}'")

    target_count = 1400  # number of sections to generate
    count = 0
    attempts = 0
    max_attempts = target_count * 10

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