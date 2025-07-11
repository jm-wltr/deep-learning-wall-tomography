import os
import ezdxf
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon, Rectangle
from PIL import Image
from shapely.geometry import Polygon as ShapelyPolygon, box
import trimesh
from trimesh.creation import extrude_polygon
from section import generate_cross_section  # or your actual module
from shapely.validation import explain_validity
from shapely.geometry import (
    Polygon, MultiPolygon,
    LineString, MultiLineString,
    Point, GeometryCollection
)
import matplotlib.pyplot as plt


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
        "size": (0.06, 0.04),
        "n_rows": 2,
        "min_div": 2,
        "max_div": 3,
        "min_width_frac": 0.2,
        "min_height_frac": 0.2,
        "TS_position": "center",
        "max_partition_attempts": 10000,
        "sides": 15,
        "K": 0.5,
        "dxf_subdir": "sectionsNormal",
        "jpg_subdir": "imagesNormal",
        "stl_subdir": "stlsNormal",
        "thickness": 0.02
    },
    "crop": {
        "full_size": (0.12, 0.04),
        "crop_size": (0.06, 0.04),
        "crop_step": 0.1,
        "n_rows": 2,
        "min_div": 4,
        "max_div": 6,
        "min_width_frac": 0.1,
        "min_height_frac": 0.2,
        "TS_position": "center",
        "max_partition_attempts": 10000,
        "sides": 15,
        "K": 0.5,
        "dxf_subdir": "sectionsCrop",
        "jpg_subdir": "imagesCrop",
        "stl_subdir": "stlsCrop",
        "thickness": 0.02
    },
    "rotate": {
        "size": (0.04, 0.06),
        "n_rows": 4,
        "min_div": 2,
        "max_div": 2,
        "min_width_frac": 0.2,
        "min_height_frac": 0.1,
        "TS_position": "none",
        "max_partition_attempts": 10000,
        "sides": 15,
        "K": 0.05,
        "dxf_subdir": "sectionsRot",
        "jpg_subdir": "imagesRotate",
        "stl_subdir": "stlsRot",
        "thickness": 0.02
    }
}

BASE_DIR = "sections_generator/output3d"

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

def export_to_stl(stones, size, out_dir, name, thickness):
    """
    Extrude each stone (a Shapely Polygon) into its own STL.
    Creates a folder named after the wall and inside exports one STL per stone,
    merging vertices to avoid duplicates.
    Returns the path to the created folder.
    """
    # Ensure the base output directory exists
    os.makedirs(out_dir, exist_ok=True)

    # Create a dedicated folder for this wall's stone STLs
    folder = os.path.join(out_dir, name)
    os.makedirs(folder, exist_ok=True)

    # Debug logging for folder creation
    print(f"[export_to_stl] Created folder: {folder}")

    for idx, poly in enumerate(stones):
        shp = ShapelyPolygon(poly)
        # Skip invalid or empty geometries
        if not shp.is_valid or shp.area == 0:
            print(f"[export_to_stl] Skipping invalid or empty stone at index {idx}")
            continue

        # Extrude to 3D mesh
        mesh3d = extrude_polygon(shp, height=thickness, engine='triangle')
        # Merge duplicate vertices for cleaner output
        mesh3d.merge_vertices()

        # Export each stone as its own STL
        stl_filename = f"{name}_stone{idx:03d}.stl"
        stl_path = os.path.join(folder, stl_filename)
        mesh3d.export(stl_path)
        print(f"[export_to_stl] Exported STL: {stl_path}")

    return folder


def export_to_stl_orig(stones, size, out_dir, name, thickness):
    """
    Extrude each stone (a Shapely Polygon) into 3D and export as one STL.
    """
    os.makedirs(out_dir, exist_ok=True)
    all_meshes = []

    for poly in stones:
        shp = ShapelyPolygon(poly)
        if not shp.is_valid or shp.area == 0:
            continue
        mesh3d = extrude_polygon(shp, height=thickness, engine='triangle')
        all_meshes.append(mesh3d)

    if not all_meshes:
        raise ValueError("No valid stone meshes to extrude.")

    combined = trimesh.util.concatenate(all_meshes)
    stl_path = os.path.join(out_dir, f"{name}.stl")
    combined.export(stl_path)
    return stl_path

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
    dxf_path = export_to_dxf(stones, size,
                              os.path.join(BASE_DIR, cfg["dxf_subdir"]),
                              name)
    jpg_path = export_to_jpg(stones, size,
                              os.path.join(BASE_DIR, cfg["jpg_subdir"]),
                              name)
    stl_path = export_to_stl(stones, size,
                              os.path.join(BASE_DIR, cfg["stl_subdir"]),
                              name, cfg["thickness"])
    return dxf_path, jpg_path, stl_path

def extract_coords(geom, out_list):
    """
    Recursively pull out coordinate rings or lines from any shapely geom.
    Polygons → exterior ring coords
    LineStrings → line coords
    Points → ignored
    GeometryCollections/Multi* → recurse
    """
    if isinstance(geom, Polygon):
        out_list.append(list(geom.exterior.coords))
    elif isinstance(geom, MultiPolygon):
        for poly in geom.geoms:
            out_list.append(list(poly.exterior.coords))
    elif isinstance(geom, LineString):
        out_list.append(list(geom.coords))
    elif isinstance(geom, MultiLineString):
        for line in geom.geoms:
            out_list.append(list(line.coords))
    elif isinstance(geom, GeometryCollection):
        for part in geom.geoms:
            extract_coords(part, out_list)
    # ignore Points and empty geometries

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

    # … export full as before …

    x = 0.0
    idx = 0
    while x <= full_size[0] - crop_w + 1e-6:
        crop_box = box(x, 0.0, x + crop_w, crop_h)
        clipped = []
        for poly in stones_full:
            shp = ShapelyPolygon(poly)
            inter = shp.intersection(crop_box)
            if inter.is_empty:
                continue
            extract_coords(inter, clipped)

        name = f"{count:05d}_crop{idx}"
        export_to_dxf( clipped, (crop_w, crop_h),
                       os.path.join(BASE_DIR, cfg["dxf_subdir"]), name )
        export_to_jpg( clipped, (crop_w, crop_h),
                       os.path.join(BASE_DIR, cfg["jpg_subdir"]), name,
                       xlim=(x, x+crop_w), ylim=(0, crop_h) )
        export_to_stl( clipped, (crop_w, crop_h),
                       os.path.join(BASE_DIR, cfg["stl_subdir"]), name, cfg["thickness"] )

        x += cfg["crop_step"]
        idx += 1

def process_rotate(cfg, count):
    size = cfg["size"]
    stones_r = generate_cross_section(
        X=size[0], Y=size[1],
        n_rows=random.randint(2, 4),
        min_div=cfg["min_div"], max_div=cfg["max_div"],
        min_width_frac=cfg["min_width_frac"],
        min_height_frac=cfg["min_height_frac"],
        TS_position=cfg["TS_position"],
        max_partition_attempts=cfg["max_partition_attempts"],
        sides=cfg["sides"],
        K=cfg["K"]
    )[-1]
    name = f"{count:05d}_rot"
    export_to_dxf(stones_r, size,
                  os.path.join(BASE_DIR, cfg["dxf_subdir"]), name)
    export_to_jpg(stones_r, size,
                  os.path.join(BASE_DIR, cfg["jpg_subdir"]), name)
    export_to_stl(stones_r, size,
                  os.path.join(BASE_DIR, cfg["stl_subdir"]), name, cfg["thickness"])

    img = Image.open(os.path.join(BASE_DIR, cfg["jpg_subdir"], f"{name}.jpg"))
    img.rotate(-90, expand=True).save(os.path.join(BASE_DIR, cfg["jpg_subdir"], f"{name}.jpg"))

def main():
    generation_mode = "crop"  # 'normal', 'crop', or 'rotate'
    cfg = MODE_CONFIGS[generation_mode]
    target_count = 700
    count, attempts, max_attempts = 0, 0, target_count * 10

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

