import json
import time
import math
from side import Side
from utils import *
from getsides import getsides
from collections import deque
from sklearn.random_projection import GaussianRandomProjection


def rot2d(theta_rad: float):
    c, s = math.cos(theta_rad), math.sin(theta_rad)
    return np.array([[c, -s],
                     [s, c]], dtype=float)


def side_direction(points):
    if len(points) < 2:
        return np.array([1.0, 0.0])
    p0 = np.array(points[0], dtype=float)
    p1 = np.array(points[-1], dtype=float)
    v = p1 - p0
    n = np.linalg.norm(v)
    if n < 1e-9:
        return np.array([1.0, 0.0])
    return v / n


def side_midpoint(points):
    if len(points) == 0:
        return np.array([0.0, 0.0])
    return np.mean(points, axis=0)


def apply_transform(pts, R, t):
    if len(pts) == 0:
        return np.array([])
    P = np.asarray(pts, dtype=float)
    if P.ndim == 1:
        P = P.reshape(1, -1)
    return (P @ R.T) + t


def compute_transform_for_bulge_trough(bulge_pts, trough_pts):
    """Compute transform to properly align bulge with trough."""
    if len(bulge_pts) < 2 or len(trough_pts) < 2:
        return np.eye(2), np.zeros(2)

    v_bulge = side_direction(bulge_pts)
    v_trough = side_direction(trough_pts)
    target_direction = -v_bulge

    dot = np.clip(np.dot(v_trough, target_direction), -1, 1)
    det = v_trough[0] * target_direction[1] - v_trough[1] * target_direction[0]
    theta = math.atan2(det, dot)
    R = rot2d(theta)

    trough_rotated = apply_transform(trough_pts, R, np.zeros(2))

    mid_bulge = side_midpoint(bulge_pts)
    mid_trough = side_midpoint(trough_rotated)

    normal = np.array([-v_bulge[1], v_bulge[0]])
    separation = 2.0

    t = mid_bulge - mid_trough + separation * normal

    return R, t


def get_piece_vertices(sides):
    vertices = []
    for side in sides:
        if len(side.side_points) > 0:
            vertices.extend(side.side_points)

    unique_vertices = []
    seen = set()
    for vertex in vertices:
        vertex_tuple = tuple(vertex)
        if vertex_tuple not in seen:
            seen.add(vertex_tuple)
            unique_vertices.append(vertex)

    return np.array(unique_vertices)


def is_valid_match(sideA, sideB, pieceA_id, pieceB_id, length_threshold=25, score_threshold=150):
    if pieceA_id == pieceB_id:
        return False
    if sideA.type == sideB.type:
        return False
    if sideA.type == 'flat' and sideB.type == 'flat':
        if hasattr(sideA, 'side_index') and hasattr(sideB, 'side_index'):
            if sideA.side_index != sideB.side_index:
                return False
    elif sideA.type == 'flat' or sideB.type == 'flat':
        return False
    return True


# -----------------------------
# Load pieces
# -----------------------------
sbs = time.time()
getsides(show=False)

with open("pieces_data.json", "r") as f:
    data = json.load(f)

pieces = {}
print(f"Loading {len(data)} pieces...")

for piece in data:
    pid = piece["piece_number"]
    sides = []
    for s in piece["sides"]:
        side = Side(s["side_points"], s["side_index"], piece_number=pid)
        side.adjust_angle()
        sides.append(side)

    polygon = get_piece_vertices(sides)

    pieces[pid] = {
        "sides": sides,
        "polygon": polygon,
        "available_sides": set(range(len(sides)))
    }

piece_ids = sorted(pieces.keys())
print(f"Loaded {len(piece_ids)} pieces: {piece_ids}")


# -----------------------------
# Fuzzy LSH clustering for bulge/trough shapes
# -----------------------------
print("Classifying sides using fuzzy LSH similarity...")

class LSHCluster:
    def __init__(self, n_components=16):
        self.proj = GaussianRandomProjection(n_components=n_components)
        self.embeddings = None
        self.sides = []

    @staticmethod
    def _baseline_transform(points):
        pts = np.array(points)
        if len(pts) < 2:
            return pts
        p1, p2 = pts[0], pts[-1]
        baseline = p2 - p1
        if np.linalg.norm(baseline) < 1e-8:
            return pts
        pts = pts - p1
        angle = -np.arctan2(baseline[1], baseline[0])
        R = np.array([[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle), np.cos(angle)]])
        aligned = pts @ R.T
        aligned[:, 0] /= aligned[-1, 0]
        return aligned

    def fit(self, sides):
        curves = []
        for s in sides:
            pts = self._baseline_transform(s.normalized_points)
            y_values = np.interp(np.linspace(0, 1, 50),
                                 np.linspace(0, 1, len(pts)), pts[:, 1])
            curves.append(y_values)
            self.sides.append(s)
        X = np.stack(curves)
        self.embeddings = self.proj.fit_transform(X)

    def group_by_similarity_fuzzy(self, base_threshold=None):
        n = len(self.embeddings)
        if base_threshold is None:
            dists = []
            for i in range(n):
                for j in range(i + 1, n):
                    dists.append(np.linalg.norm(self.embeddings[i] - self.embeddings[j]))
            base_threshold = np.mean(dists) + np.std(dists)

        clusters = []
        for i in range(n):
            group = []
            for j in range(n):
                if np.linalg.norm(self.embeddings[i] - self.embeddings[j]) <= base_threshold:
                    group.append(j)
            clusters.append(group)
        return clusters


flat_sides = []
bulge_sides = []
trough_sides = []

for pid, pdata in pieces.items():
    for s in pdata["sides"]:
        if s.type == "flat":
            flat_sides.append(s)
        elif s.type == "outward":
            bulge_sides.append(s)
        elif s.type == "inward":
            trough_sides.append(s)

# Fit LSH to curved sides
if bulge_sides or trough_sides:
    lsh = LSHCluster(n_components=16)
    lsh.fit(bulge_sides + trough_sides)
    clusters = lsh.group_by_similarity_fuzzy()
    print(f"Formed {len(clusters)} fuzzy LSH shape clusters for bulge/trough sides.")
else:
    print("No curved sides found for LSH clustering.")
    clusters = []

# Map sides to fuzzy clusters
side_to_clusters = {}
for cluster_idx, cluster in enumerate(clusters):
    for side_idx in cluster:
        sid = lsh.sides[side_idx].side_id
        if sid not in side_to_clusters:
            side_to_clusters[sid] = []
        side_to_clusters[sid].append(cluster_idx)

# -----------------------------
# Compute candidate matches using fuzzy LSH clusters
# -----------------------------
print("Computing candidate matches using fuzzy LSH clusters...")
candidates = []

for cluster_idx, cluster in enumerate(clusters):
    cluster_bulges = [s for s_idx in cluster if (s := lsh.sides[s_idx]).type == 'outward']
    cluster_troughs = [s for s_idx in cluster if (s := lsh.sides[s_idx]).type == 'inward']

    for sideA in cluster_bulges:
        for sideB in cluster_troughs:
            if sideA.piece_number == sideB.piece_number:
                continue
            if not is_valid_match(sideA, sideB, sideA.piece_number, sideB.piece_number):
                continue
            try:
                score, best_shift = matching_score(sideA, sideB)
                candidates.append((score,
                                   sideA.piece_number, sideA.side_index,
                                   sideB.piece_number, sideB.side_index,
                                   best_shift))
            except Exception as e:
                print(f"Error scoring {sideA.piece_number}-{sideA.side_index} with {sideB.piece_number}-{sideB.side_index}: {e}")
                continue

candidates.sort(key=lambda x: x[0])
print(f"Found {len(candidates)} candidate matches using fuzzy clusters")


# -----------------------------
# BFS placement with transformations
# -----------------------------
placements = {}
matched_sides = {}
unplaced = set(piece_ids)
connection_log = []

anchor = 0
placements[anchor] = (np.eye(2), np.zeros(2))
unplaced.remove(anchor)
queue = deque([anchor])

print("\nStarting puzzle assembly...")

while queue and unplaced:
    current_piece = queue.popleft()
    current_R, current_t = placements[current_piece]

    print(f"Processing piece {current_piece} (queue: {len(queue)}, unplaced: {len(unplaced)})")

    best_match = None
    best_score = float('inf')

    for candidate in candidates:
        score, A, a_idx, B, b_idx, shift = candidate

        if A != current_piece and B != current_piece:
            continue

        if A == current_piece:
            neighbor, nbr_side_idx = B, b_idx
            curr_side_idx = a_idx
        else:
            neighbor, nbr_side_idx = A, a_idx
            curr_side_idx = b_idx

        if neighbor not in unplaced:
            continue
        if curr_side_idx not in pieces[current_piece]["available_sides"]:
            continue
        if nbr_side_idx not in pieces[neighbor]["available_sides"]:
            continue

        if score < best_score:
            best_score = score
            best_match = (score, current_piece, curr_side_idx, neighbor, nbr_side_idx, shift)

    if best_match is None:
        print(f"  No valid matches found for piece {current_piece}")
        continue

    score, curr_piece, curr_side_idx, neighbor, nbr_side_idx, shift = best_match

    print(f"  Best match: piece {curr_piece} side {curr_side_idx} -> piece {neighbor} side {nbr_side_idx} (score: {score:.2f})")

    try:
        curr_side = pieces[curr_piece]["sides"][curr_side_idx]
        nbr_side = pieces[neighbor]["sides"][nbr_side_idx]

        if curr_side.type == 'outward' and nbr_side.type == 'inward':
            bulge_pts, trough_pts = curr_side.side_points, nbr_side.side_points
            bulge_transformed = apply_transform(bulge_pts, current_R, current_t)
            R_rel, t_rel = compute_transform_for_bulge_trough(bulge_transformed, trough_pts)
        else:
            trough_pts, bulge_pts = curr_side.side_points, nbr_side.side_points
            trough_transformed = apply_transform(trough_pts, current_R, current_t)
            R_rel, t_rel = compute_transform_for_bulge_trough(bulge_pts, trough_transformed)
            R_rel = R_rel.T
            t_rel = -t_rel

        R_world = R_rel @ current_R
        t_world = (R_rel @ current_t) + t_rel

        placements[neighbor] = (R_world, t_world)

        matched_sides[(curr_piece, curr_side_idx)] = (neighbor, nbr_side_idx)
        matched_sides[(neighbor, nbr_side_idx)] = (curr_piece, curr_side_idx)
        connection_log.append((curr_piece, curr_side_idx, neighbor, nbr_side_idx))

        pieces[curr_piece]["available_sides"].discard(curr_side_idx)
        pieces[neighbor]["available_sides"].discard(nbr_side_idx)

        unplaced.remove(neighbor)
        queue.append(neighbor)

    except Exception as e:
        print(f"  ERROR transforming piece {neighbor}: {e}")
        continue

if unplaced:
    print(f"\nPlacing {len(unplaced)} unplaced pieces around the assembly...")
    base_offset = len(placements) * 100
    for i, pid in enumerate(unplaced):
        angle = (i * 2 * math.pi) / len(unplaced)
        offset_x = base_offset * math.cos(angle)
        offset_y = base_offset * math.sin(angle)
        placements[pid] = (np.eye(2), np.array([offset_x, offset_y]))

elapsed = time.time() - sbs
print(f"\n=== ASSEMBLY COMPLETE ===")
print(f"Time: {elapsed:.2f}s")
print(f"Pieces placed: {len(placements)}/{len(piece_ids)}")
print(f"Connections made: {len(connection_log)}")
print(f"Connection log:")
for conn in connection_log:
    print(f"  Piece {conn[0]} side {conn[1]} -> Piece {conn[2]} side {conn[3]}")
