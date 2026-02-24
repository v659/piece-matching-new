from utils import *
import matplotlib

import matplotlib.pyplot as plt
import numpy as np
from side import Side
from shapely.geometry import Polygon, MultiPolygon, LineString

def plot_geometries(geometries):
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_title("Visualized Geometries")
    for geom in geometries:
        if isinstance(geom, Polygon):
            x, y = geom.exterior.xy
            ax.plot(x, y, 'b')
        elif isinstance(geom, MultiPolygon):
            for poly in geom.geoms:
                x, y = poly.exterior.xy
                ax.plot(x, y, 'g')
        else:
            print(f"️ Skipping unknown geometry type: {type(geom)}")

    ax.invert_yaxis()
    plt.tight_layout()
    plt.show()
def getsides(show=True):
    import json
    piece_count = 0  # Label for each piece
    binarized, img_np = binarize_image(image)

    fig_bin, ax_bin = plt.subplots()
    ax_bin.set_title("Binarized Image")
    ax_bin.imshow(binarized, cmap='gray')
    ax_bin.axis('off')
    fig_bin.tight_layout()
    fig_bin.savefig("binarized_image.png")

    img_draw = np.array(image.convert("RGB"))  # For visualization
    boxes = get_blobs(img_np, draw_on=img_draw)
    geometries = get_edge(img_np, boxes, sample_every=5)
    print(f"Extracted {len(geometries)} geometries")
    # === Plotting ===
    fig, ax = plt.subplots()
    fig2, ax2 = plt.subplots()
    ax.set_aspect('equal')
    ax2.set_aspect('equal')
    ax.set_title("Piece Shapes with Labels")
    ax2.set_title("Detected Sides with Index Labels")
    fig3, ax3 = plt.subplots()
    ax3.set_title("All Rotated Sides")
    ax3.set_aspect('equal')
    ax3.grid(True)
    ax3.legend(fontsize=6)

    sides_list = []
    sides_list_copy = []
    with open("pieces_data.json", "w") as f:
        json.dump([], f)
    with open("types_data.json", "w") as f:
        json.dump([], f)
    for geom in geometries:

        if isinstance(geom, Polygon):
            polygons = [geom]
        elif isinstance(geom, MultiPolygon):
            polygons = list(geom.geoms)
        elif isinstance(geom, LineString):
            coords = list(geom.coords)
            x_vals = [pt[0] for pt in coords]
            y_vals = [pt[1] for pt in coords]
            continue
        else:
            print(f"⚠️ Skipping unknown geometry type: {type(geom)}")
            continue

        for poly in polygons:
            piece_count += 1
            x, y = poly.exterior.xy
            ax.plot(x, y, color='black', linewidth=2)
            center = poly.representative_point()
            ax.text(
                center.x,
                center.y,
                str(piece_count),
                fontsize=10,
                color='darkred',
                ha='center',
                va='center',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
            )

            sides, candidate_corners, side_index_dict = detect_polygon_corners_by_rdp(poly, ax2)
            for idx, pt in candidate_corners:
                ax2.plot(pt[0], pt[1], 'ro', markersize=4)

            if not sides:
                print("❌ No valid quad for this polygon")
                continue
            cmap = plt.get_cmap('tab20')
            for i in side_index_dict:
                side_arr = np.array(side_index_dict[i])
                side_arr_copy = rotate_points_to_nearest_right_angle(side_arr)
                if len(side_arr) < 2:
                    continue

                side_obj = Side(list(tuple(i) for i in side_arr.tolist()), i)
                side_obj_copy = Side(list(tuple(i) for i in side_arr_copy.tolist()), i)
                sides_list_copy.append(side_obj_copy)
                sides_list.append(side_obj)


                color = cmap(i % 20)

                ax2.plot(side_arr[:, 0], side_arr[:, 1], color=color, linewidth=2)

                mid_idx = len(side_arr) // 2
                mid_pt = side_arr[mid_idx]
                label = str(side_obj.side_index)

                ax2.text(mid_pt[0], mid_pt[1], label, fontsize=8, color='blue',
                         ha='center', va='center',
                         bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

    sides_list = split_list(sides_list)
    figures = []
    axes = []
    with open("pieces_data.json", "r") as f:
        data = json.load(f)
    with open("types_data.json", "r") as f:
        data2 = json.load(f)
    for j, i in enumerate(sides_list_copy):
        side_arr2 = np.array(i.side_points)
        cmap = plt.get_cmap('tab20')
        color = cmap(j % 20)

        side_type = classify_side_shape(side_arr2, ax=ax3)

        ax3.plot(
            side_arr2[:, 0],
            side_arr2[:, 1],
            color=color
        )
        mid_pt = side_arr2[len(side_arr2) // 2]
        ax3.text(mid_pt[0], mid_pt[1], str(i.side_index), fontsize=8)
        # Endpoints and baseline
        p1 = side_arr2[0]
        p2 = side_arr2[-1]
        ax3.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k--', linewidth=1)  # baseline (dashed black)

        # Midpoint of the side
        mid_pt = side_arr2[len(side_arr2) // 2]

        # Projection of midpoint onto baseline
        line_vec = p2 - p1
        line_len = np.linalg.norm(line_vec)
        if line_len > 0:
            line_unit = line_vec / line_len
            vec = mid_pt - p1
            proj_len = np.dot(vec, line_unit)
            closest = p1 + proj_len * line_unit



        data2.append(side_type)

    ax3.set_title("All Rotated Sides")
    ax3.set_aspect('equal')
    ax3.legend(fontsize=6)
    fig3.tight_layout()

    with open("types_data.json", "w") as f:
        json.dump(data2, f, indent=4)
    for i, piece_sides in enumerate(sides_list):
        piece_data = {
            "piece_number": int(i),
            "sides": []
        }

        for r in piece_sides:

            side_data = {
                "side_points": [list(map(float, pt)) for pt in r.side_points],
                "side_index": int(r.side_index),
                "side_length": float(r.length),
                "side_angle": float(r.angle),
                "side_normalized": [list(map(float, pt)) for pt in r.normalized_points] if r.normalized_points is not None
 else [],
                "p1": list(map(float, r.p1)),
                "p2": list(map(float, r.p2)),
                "side_id": r.side_id,
                "side_data": str(r)
            }
            piece_data["sides"].append(side_data)

        data.append(piece_data)

        print(f"\n{'Piece':<8} {'Side Lengths (true px)':<45} {'Sum'} {'Type':<20} ")
        print("-" * 80)

        side_offset = 0

        for i, piece_sides in enumerate(sides_list):
            lengths = [round(r.path_length) for r in piece_sides]
            total_length = sum(lengths)
            piece_side_types = data2[side_offset:side_offset + len(piece_sides)]
            piece_type = str(piece_side_types)

            print(f"{i:<8} {str(lengths):<45} {total_length:.2f} {piece_type:<20}")

            side_offset += len(piece_sides)

    with open("pieces_data.json", "w") as f:
        json.dump(data, f, indent=4)

    fig.tight_layout()
    fig2.tight_layout()
    fig.gca().invert_yaxis()
    fig2.gca().invert_yaxis()
    fig3.gca().invert_yaxis()
    print('done')
    fig.savefig("piece_shapes.png")
    fig2.savefig("side_segments.png")
    fig3.savefig("classified")
    if show:
        plt.show(block=True)


if __name__ == "__main__":
    getsides()
