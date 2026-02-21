import os
import sys
from collections import defaultdict
from itertools import combinations
import matplotlib.pyplot as plt
import cv2
import numpy as np
import shapely
from PIL import Image
from rdp import rdp
from scipy.interpolate import splprep, splev
from scipy.spatial.distance import directed_hausdorff
from shapely.geometry import Polygon, LineString


def resource_path(relative_path):
    try:
        # When runnng from the exe
        base_path = sys._MEIPASS
    except AttributeError:
        # When running from the .py file
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


# Example usage
image_path = resource_path("IMG_9868.jpg")
image = Image.open(image_path)


def find_array_with_points(sides, pts):
    """Returns one array in a array of sides which has the given points"""
    pt1 = np.array(pts[0])
    pt2 = np.array(pts[1])
    for arr in sides:
        has_pt1 = np.any(np.all(arr == pt1, axis=1))
        has_pt2 = np.any(np.all(arr == pt2, axis=1))
        if has_pt1 and has_pt2:
            return arr
    raise ValueError("No array contains both points.")


def binarize_image(img, thresh=100, erode_iterations=2, kernel_size=3):
    img = img.convert('L')
    img_np = np.array(img)
    binarized_np = (img_np > thresh).astype(np.uint8) * 255

    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    eroded_np = cv2.erode(binarized_np, kernel, iterations=erode_iterations)

    return Image.fromarray(eroded_np), eroded_np // 255


def get_blobs(binarized_np, min_area=50000, draw_on=None):
    binary = (binarized_np * 255).astype(np.uint8) if binarized_np.max() <= 1 else binarized_np.copy()
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for contour in contours:
        # Skip small areas
        if cv2.contourArea(contour) < min_area:
            continue
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.intp(box)

        boxes.append(box)

        if draw_on is not None:
            cv2.drawContours(draw_on, [box], 0, (0, 255, 0), 2)

    print(f"Detected {len(boxes)} rotated blobs.")
    return boxes


# noinspection PyTypeChecker
def get_edge(binarized_np, boxes, sample_every=5):
    geometries = []

    for box in boxes:
        mask = np.zeros_like(binarized_np, dtype=np.uint8)
        cv2.drawContours(mask, [box], 0, color=255, thickness=-1)

        blob = cv2.bitwise_and(binarized_np, mask)

        cnts, _ = cv2.findContours(blob, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
        if cnts:
            largest = max(cnts, key=cv2.contourArea)
            largest = largest.reshape(-1, 2)
            sampled = largest[::sample_every]

            if len(sampled) >= 3:
                poly = Polygon(sampled)
                print(f"Box - Contour points: {len(sampled)} | Valid: {poly.is_valid} | Area: {poly.area:.2f}")
                if not poly.is_valid:
                    print("Invalid polygon — attempting fix...")
                    poly = poly.buffer(0)

                if poly.is_valid:
                    geometries.append(poly)
            elif len(sampled) >= 2:
                geometries.append(LineString(sampled))

    return geometries


def get_centroid(polygon):
    """Returns gripping point(centroid) of each piece(polygon)"""
    return shapely.centroid(polygon)


def get_polygon_size(polygon):
    """Returns dimension of a piece(polygon)"""
    minx, miny, maxx, maxy = polygon.bounds
    width = maxx - minx
    height = maxy - miny
    return height, width


def label_box_corners(points):
    points = np.array(points)

    sorted_by_y = points[np.argsort(points[:, 1])]

    top = sorted_by_y[2:]
    bottom = sorted_by_y[:2]

    top_left, top_right = top[np.argsort(top[:, 0])]
    bottom_left, bottom_right = bottom[np.argsort(bottom[:, 0])]

    return {
        1: tuple(top_left),
        2: tuple(bottom_left),
        3: tuple(bottom_right),
        4: tuple(top_right)
    }


def get_side_corners(combo):
    """Construct the side from the sorted corners"""
    side_ends = [
        [combo[1], combo[2]],
        [combo[2], combo[3]],
        [combo[3], combo[4]],
        [combo[4], combo[1]]
    ]
    return side_ends


def isParallel(line1, line2, visualize=True):
    p1, p2 = line1
    p3, p4 = line2

    v1 = p2 - p1
    v2 = p4 - p3

    cross = np.cross(v1, v2)
    parallel = np.isclose(cross, 0.0, atol=1e-6)

    if visualize:
        plt.figure(figsize=(5, 5))
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-', label='Line 1')
        plt.plot([p3[0], p4[0]], [p3[1], p4[1]], 'r-', label='Line 2')
        plt.scatter([p1[0], p2[0], p3[0], p4[0]],
                    [p1[1], p2[1], p3[1], p4[1]], color='blue')
        plt.title(f"Parallel: {parallel}")
        plt.axis('equal')
        plt.legend()
        plt.grid(True)
        plt.show()

    return bool(parallel)


def detect_polygon_corners_by_rdp(polygon, ax2=None, epsilon=50, min_length=None, ax=None):
    coords = np.array(polygon.exterior.coords[:-1])
    simplified = rdp(coords, epsilon=epsilon)
    candidate_corners = list(enumerate(simplified))
    print(f"  Candidate count: {len(candidate_corners)}")
    if ax:
        for idx, pt in candidate_corners:
            ax.plot(pt[0], pt[1], 'ro', markersize=3)

    if len(candidate_corners) < 4:
        return [], candidate_corners, {}

    perimeter = np.sum(np.linalg.norm(np.diff(coords, axis=0), axis=1))
    if min_length is None:
        min_length = perimeter * 0.12

    rejected_spacing = 0
    rejected_zero_sides = 0
    rejected_min_length = 0
    top_scores = []

    def corner_angle_error_at_index(idx_in_coords, coords_arr, window=1):
        n = len(coords_arr)
        before_idx = (idx_in_coords - window) % n
        after_idx = (idx_in_coords + window) % n
        p_before = coords_arr[before_idx]
        p_corner = coords_arr[idx_in_coords]
        p_after = coords_arr[after_idx]
        v1 = p_before - p_corner
        v2 = p_after - p_corner
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-8 or n2 < 1e-8:
            return 90.
        cosang = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
        angle = np.degrees(np.arccos(cosang))
        return abs(angle - 90.0)

    best_combo = None
    best_score = float('inf')
    best_between_counts = []
    best_true_lengths = []

    for combo in combinations(candidate_corners, 4):
        points = [pt for _, pt in combo]
        indices = [idx for idx, _ in combo]

        index_to_position = {idx: pos for pos, (idx, _) in enumerate(candidate_corners)}
        boundary_positions = [index_to_position[idx] for idx in indices]

        # --- SPACING CHECK ---
        valid_spacing = True
        current_between_counts = []
        for i in range(4):
            start_pos = boundary_positions[i]
            end_pos = boundary_positions[(i + 1) % 4]

            if start_pos < end_pos:
                between_count = end_pos - start_pos - 1
            else:
                between_count = len(candidate_corners) - start_pos + end_pos - 1

            current_between_counts.append(between_count)

            if 0 < between_count < 3:
                valid_spacing = False
                break

        if not valid_spacing:
            rejected_spacing += 1
            continue

        zero_sides = sum(1 for b in current_between_counts if b == 0)
        if zero_sides > 2:
            rejected_zero_sides += 1
            continue

        sorted_points = sorted(
            points, key=lambda p: np.arctan2(p[1] - polygon.centroid.y, p[0] - polygon.centroid.x)
        )
        sorted_indices = []
        for sorted_pt in sorted_points:
            for i, orig_pt in enumerate(points):
                if np.allclose(sorted_pt, orig_pt):
                    sorted_indices.append(indices[i])
                    break

        rdp_indices = [np.argmin(np.linalg.norm(coords - corner, axis=1)) for corner in sorted_points]

        # --- TRUE LENGTH CALCULATIONS ---
        true_lengths = []
        for i in range(4):
            start = rdp_indices[i]
            end = rdp_indices[(i + 1) % 4]
            if start < end:
                segment = coords[start:end + 1]
            else:
                segment = np.concatenate((coords[start:], coords[:end + 1]), axis=0)
            seg_length = np.sum(np.linalg.norm(np.diff(segment, axis=0), axis=1))
            true_lengths.append(seg_length)

        if any(l < min_length for l in true_lengths):
            rejected_min_length += 1
            continue

        angle_errors = [corner_angle_error_at_index(idx, coords, window=1) for idx in rdp_indices]
        mean_angle_error = np.mean(angle_errors)

        mean_len = np.mean(true_lengths)
        std_len = np.std(true_lengths)
        length_ratio = std_len / (mean_len + 1e-8)

        min_len = min(true_lengths)
        max_len = max(true_lengths)
        shortness_penalty = 1.0 - (min_len / (max_len + 1e-8))

        score = mean_angle_error + length_ratio * 50 + shortness_penalty * 100

        top_scores.append((score, [round(l) for l in true_lengths], current_between_counts))

        if score < best_score:
            best_score = score
            best_combo = sorted_points
            best_rdp_indices = rdp_indices
            best_between_counts = current_between_counts
            best_true_lengths = true_lengths

    print(f"  Rejected - spacing: {rejected_spacing} | zero_sides: {rejected_zero_sides} | min_length: {rejected_min_length}")
    top_scores.sort(key=lambda x: x[0])
    print(f"  Top 5 scoring combos:")
    for score, lengths, bc in top_scores[:5]:
        print(f"    score={score:.2f}  lengths={lengths}  between={bc}")

    if best_combo is None:
        print("❌ No valid quad for this polygon")
        return [], candidate_corners, {}

    sorted_pairs = sorted(zip(best_rdp_indices, best_combo), key=lambda x: x[0])
    ordered_indices = [idx for idx, _ in sorted_pairs]
    validated_side_points = []
    for i in range(4):
        start = ordered_indices[i]
        end = ordered_indices[(i + 1) % 4]
        if start < end:
            segment = coords[start:end + 1]
        else:
            segment = np.concatenate((coords[start:], coords[:end + 1]), axis=0)
        if (np.linalg.norm(segment[0] - coords[start]) > 1e-3 or
                np.linalg.norm(segment[-1] - coords[end]) > 1e-3):
            print(f"Skipping side {i} due to improper endpoints")
            continue
        validated_side_points.append(np.array(segment))
    side_points = validated_side_points

    point_to_sides = defaultdict(list)
    for side_idx, side in enumerate(side_points):
        for pt in map(tuple, side):
            point_to_sides[pt].append(side_idx)

    for pt, sides in point_to_sides.items():
        if len(sides) > 1 and ax2:
            ax2.plot(pt[0], pt[1], 'yx', markersize=10)

    labeled = label_box_corners(best_combo)
    side_index_dict = {}
    for j, i in enumerate(get_side_corners(labeled)):
        side_index_dict[j] = find_array_with_points(side_points, i)

    print(f"  Best combo between_counts: {best_between_counts}")
    print(f"  Best score: {best_score:.2f}  lengths={[round(l) for l in best_true_lengths]}")
    return side_points, candidate_corners, side_index_dict


def classify_side_shape(obj, ax=None, flat_threshold=100):
    """Classifies the type of shape"""
    # Handle the Side object or numpy array
    if hasattr(obj, "side_points"):
        points = np.array(obj.side_points)
    else:
        points = np.array(obj)

    if len(points) < 3:
        return "flat"

    # Endpoints = baseline
    p1, p2 = points[0], points[-1]
    line_vec = p2 - p1
    line_len = np.linalg.norm(line_vec)
    if line_len < 1e-8:
        return "flat"
    line_unit = line_vec / line_len

    # Midpoint of a curve
    mid = points[len(points) // 2]

    # Projection of midpoint onto baseline
    vec = mid - p1


    # Signed deviation (2D cross product gives orientation)
    signed_dist = np.cross(line_unit, vec)
    abs_dist = abs(signed_dist)

    # Classification
    if abs_dist < flat_threshold:
        side_type = "flat"
    elif signed_dist > 0:
        side_type = "outward"
    else:
        side_type = "inward"

    return side_type


# noinspection PyTypeChecker,PyTupleAssignmentBalance
def normalized(side, num_points=50):
    """Normalizes the side for further comparisons by spacing it and aligning axes to 0"""
    points = np.array(side.side_points)
    if len(points) <= 3:
        print(f" Not enough points to interpolate (got {len(points)} points). Skipping normalization.")
        return points

    try:
        x, y = points[:, 0], points[:, 1]
        tck, _ = splprep([x, y], s=0)
        u = np.linspace(0, 1, num_points)
        x_i, y_i = splev(u, tck)
        curve = np.stack([x_i, y_i], axis=1)

        current_angle = side.angle
        rotation_deg = -current_angle
        rotated_curve = rotate_points(curve, origin=side.p1, angle_deg=rotation_deg)

        rotated_curve[:, 0] -= rotated_curve[0, 0]

        rotated_curve[:, 1] -= np.mean(rotated_curve[:, 1])

        return rotated_curve

    except Exception as e:
        print(f"❌ splprep failed on side: {e}")
        return points


def matching_score(side_a, side_b):
    """Outputs how well 2 sides match"""
    norm_a = normalized(side_a)
    norm_b = normalized(side_b)

    ref_point = np.array(side_a.p1)
    norm_a += ref_point - norm_a[0]  # shift side_a's curve so p1 aligns to real p1

    variations = {
        "original": norm_b,
        "flip_x": norm_b * np.array([-1, 1]),
        "flip_y": norm_b * np.array([1, -1]),
        "flip_xy": norm_b * np.array([-1, -1]),
    }

    best_score = float("inf")
    best_points = norm_b

    for mode, variant in variations.items():
        variant_shifted = variant - variant[0] + ref_point

        forward = directed_hausdorff(norm_a, variant_shifted)[0]
        backward = directed_hausdorff(variant_shifted, norm_a)[0]
        score = max(forward, backward)

        # print(f"Score ({mode}): {score:.2f}")
        if score < best_score:
            best_score = score
            best_points = variant_shifted.copy()

    return best_score, best_points


def sanity_check_polygon(polygon):
    if not polygon.is_valid:
        print("Invalid polygon geometry!")
        return False
    if polygon.area == 0:
        print("Zero area polygon!")
        return False
    if len(polygon.exterior.coords) < 4:
        print("Too few points for a valid shape!")
        return False
    return True


def side_to_dict(p1, p2, side_index, side_points):
    return {
        "p1": p1.tolist(),
        "p2": p2.tolist(),
        "side_index": side_index,
        "points": list(tuple(i) for i in side_points.tolist()),
    }


# used some AI to explain this one
def rotate_points(points, origin, angle_deg):
    angle_rad = np.radians(angle_deg)
    rot_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad), np.cos(angle_rad)]
    ])
    return (points - origin) @ rot_matrix.T + origin


# noinspection PyPep8Naming
def rotate_points_to_nearest_right_angle(points):
    # Use PCA to find the main axis direction
    centered = points - np.mean(points, axis=0)
    u, s, vh = np.linalg.svd(centered)
    direction = vh[0]  # principal component direction

    angle = np.degrees(np.arctan2(direction[1], direction[0]))
    target = min([0, 90, 180, -90], key=lambda x: abs((x - angle + 180) % 360 - 180))
    theta = np.radians((target - angle + 180) % 360 - 180)

    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    return centered @ R.T + np.mean(points, axis=0)


def split_list(lst, chunk_size=4):
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def orient_sides_from_corners(corners):
    """
    Given 4 corners (unsorted), return ordered sides as a list of point arrays.
    Side 1 is always the right side.
    """

    corners = np.array(corners)

    # Step 1: Sort corners by y to get top/bottom
    sorted_by_y = corners[np.argsort(corners[:, 1])]
    top_two = sorted_by_y[:2]
    bottom_two = sorted_by_y[2:]

    # Step 2: Sort left/right within each
    top_left, top_right = top_two[np.argsort(top_two[:, 0])]
    bottom_left, bottom_right = bottom_two[np.argsort(bottom_two[:, 0])]

    # Now assemble corners in clockwise order starting from top-left
    ordered_corners = [top_left, top_right, bottom_right, bottom_left]

    # Step 3: Extract 4 sides from consecutive corners
    sides = []
    for i in range(4):
        p1 = ordered_corners[i]
        p2 = ordered_corners[(i + 1) % 4]
        side = np.array([p1, p2])
        sides.append(side)

    return sides  # [top, right, bottom, left] clockwise
