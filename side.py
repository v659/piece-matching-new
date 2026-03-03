from utils import normalized, rotate_points, classify_side_shape, side_to_dict
import numpy as np
import math
import hashlib
from rdp import rdp


class Side:
    def __init__(self, side_points, side_index, piece_number=None):
        self.side_points = rdp(np.array(side_points), epsilon=0)
        self.side_index = side_index
        self.p1 = self.side_points[0]
        self.p2 = self.side_points[-1]
        self.side_id = self.generate_id()
        self.piece_number = piece_number

    def generate_id(self):
        point_str = ','.join(map(lambda x: f"{x:.2f}", self.side_points.flatten()))
        base_str = f"{self.side_index}-{point_str}"
        return hashlib.sha1(base_str.encode()).hexdigest()[:8]

    @staticmethod
    def compute_id_from(index, points):
        flat_points = np.array(points).flatten()
        point_str = ','.join(map(lambda x: f"{x:.2f}", flat_points))
        base_str = f"{index}-{point_str}"
        return hashlib.sha1(base_str.encode()).hexdigest()[:8]

    def __str__(self):
        return f"Side(ID={self.side_id}, Index={self.side_index}, Length={self.length:.2f}, " \
               f"Angle={self.angle:.2f}°, Type={self.type})"


    def straight_line_points(self, n=50):
        start = np.array(self.p1)
        end = np.array(self.p2)
        t = np.linspace(0, 1, n)[:, None]
        return (1 - t) * start + t * end

    @property
    def straight_line(self):
        return self.straight_line_points(50)




    def adjust_angle(self, target_angle_deg=None):
        dx = self.p2[0] - self.p1[0]
        dy = self.p2[1] - self.p1[1]
        current_angle = math.degrees(math.atan2(dy, dx))

        if target_angle_deg is None:
            if self.side_index in (0, 2):
                target_angle_deg = 0
            elif self.side_index in (1, 3):
                target_angle_deg = 90
            else:
                return

        rotation_deg = target_angle_deg - current_angle

        self.side_points = rotate_points(
            self.side_points, origin=self.p1, angle_deg=rotation_deg
        )

        self.p1 = self.side_points[0]
        self.p2 = self.side_points[-1]

    @property
    def axis(self):
        if self.side_index in (0, 2):
            return "vertical"
        elif self.side_index in (1, 3):
            return "horizontal"

    @property
    def length(self):
        return np.linalg.norm(np.array(self.p1) - np.array(self.p2))

    @property
    def path_length(self):
        pts = np.array(self.side_points)
        if len(pts) < 2:
            return 0.0
        return np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))

    @property
    def angle(self):
        dx = self.p2[0] - self.p1[0]
        dy = self.p2[1] - self.p1[1]
        return math.degrees(math.atan2(dy, dx))

    @property
    def type(self):
        return classify_side_shape(self)

    @property
    def normalized_points(self):
        return normalized(self, num_points=50)

    @property
    def rdp_version(self):
        return rdp(self, epsilon=1)

    @property
    def x(self):
        if len(self.side_points) == 0:
            return (self.p1[0] + self.p2[0]) / 2
        mid_idx = len(self.side_points) // 2
        return self.side_points[mid_idx][0]

    @property
    def y(self):
        if len(self.side_points) == 0:
            return (self.p1[1] + self.p2[1]) / 2
        mid_idx = len(self.side_points) // 2
        return self.side_points[mid_idx][1]

    @property
    def midpoint(self):
        return (self.x, self.y)

    def to_dict(self):
        return side_to_dict(self.p1, self.p2, self.side_index, self.side_points)