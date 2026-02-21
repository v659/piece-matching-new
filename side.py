from utils import *
import matplotlib.pyplot as plt
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
        """
        Recompute the hash-based ID from index and points.
        This is useful to 'decode' or validate a side_id.
        """
        flat_points = np.array(points).flatten()
        point_str = ','.join(map(lambda x: f"{x:.2f}", flat_points))
        base_str = f"{index}-{point_str}"
        return hashlib.sha1(base_str.encode()).hexdigest()[:8]

    def __str__(self):
        return f"Side(ID={self.side_id}, Index={self.side_index}, Length={self.length:.2f}, " \
               f"Angle={self.angle:.2f}°, Type={self.type})"

    def adjust_angle(self, target_angle_deg=None):
        """
        Rotate the side so its direction aligns with `target_angle_deg`.
        If no target is provided, defaults to 0° (horizontal) for sides 0,2 and 90° (vertical) for sides 1,3.
        """
        dx = self.p2[0] - self.p1[0]
        dy = self.p2[1] - self.p1[1]
        current_angle = math.degrees(math.atan2(dy, dx))

        if target_angle_deg is None:
            # Use default axis-alignment if target not given
            if self.side_index in (0, 2):  # vertical edges
                target_angle_deg = 0
            elif self.side_index in (1, 3):  # horizontal edges
                target_angle_deg = 90
            else:
                return  # Unknown index

        rotation_deg = target_angle_deg - current_angle
        # print(f"Rotating side {self.side_index}:")
        # print(f"  Current angle: {current_angle:.2f}°")
        # print(f"  Target angle:  {target_angle_deg:.2f}°")
        # print(f"  Rotate by:     {rotation_deg:.2f}°")

        # Rotate the entire shape around p1
        self.side_points = rotate_points(self.side_points, origin=self.p1, angle_deg=rotation_deg)

        # Update p1 and p2
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
        """Straight-line distance between endpoints"""
        return np.linalg.norm(np.array(self.p1) - np.array(self.p2))

    @property
    def path_length(self):
        """True length along the contour (sum of consecutive point distances)"""
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


# noinspection PyTypeChecker
def visualize_side_rotation(before, after, old_angle_deg, rotation_deg, new_angle_deg):
    before = np.array(before)
    after = np.array(after)

    plt.figure(figsize=(8, 6))
    plt.plot(before[:, 0], before[:, 1], 'o--', label=f'Before (angle = {old_angle_deg:.2f}°)', color='red')
    for i, pt in enumerate(before):
        plt.text(pt[0], pt[1] + 0.05, f"B{i}", color='red', fontsize=9)

    plt.plot(after[:, 0], after[:, 1], 'o--', label=f'After (angle = {new_angle_deg:.2f}°)', color='green')
    for i, pt in enumerate(after):
        plt.text(pt[0], pt[1] - 0.1, f"A{i}", color='green', fontsize=9)

    plt.title(f"Rotation: {rotation_deg:.2f}°")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.gca().invert_yaxis()  # Flip Y-axis to match the image coordinate system

    plt.axhline(0, color='gray', lw=0.5)
    plt.axvline(0, color='gray', lw=0.5)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    plt.grid(True)
    plt.show()


def visualize_single_side(side, title="Side Visualization"):
    points = np.array(side.side_points)

    # Plot points
    plt.figure(figsize=(6, 3))
    plt.plot(points[:, 0], points[:, 1], 'bo-', label='Side Shape')
    plt.scatter(points[0, 0], points[0, 1], color='green', label='Start')
    plt.scatter(points[-1, 0], points[-1, 1], color='red', label='End')

    # Plot angle line
    angle_rad = np.deg2rad(side.angle)  # convert degrees to radians
    dx = np.cos(angle_rad)
    dy = np.sin(angle_rad)
    center = points.mean(axis=0)
    plt.arrow(center[0], center[1], dx, dy, color='orange', width=0.05, label="Angle")

    plt.title(f"{title}\nAngle: {side.angle:.6f}° | Type: {side.type}")
    plt.gca().invert_yaxis()  # Flip Y-axis to match the image coordinate system

    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    points = [(np.float64(600.0), np.float64(437.0)), (np.float64(604.0), np.float64(439.0)), (np.float64(609.0),
              np.float64(440.0)), (np.float64(614.0), np.float64(440.0)), (np.float64(619.0), np.float64(441.0)),
              (np.float64(624.0), np.float64(441.0)), (np.float64(629.0), np.float64(442.0)),
              (np.float64(634.0), np.float64(442.0)), (np.float64(639.0), np.float64(442.0)),
              (np.float64(644.0), np.float64(442.0)), (np.float64(649.0), np.float64(442.0)),
              (np.float64(654.0), np.float64(442.0)), (np.float64(659.0), np.float64(442.0)),
              (np.float64(664.0), np.float64(442.0)), (np.float64(669.0), np.float64(442.0)),
              (np.float64(674.0), np.float64(442.0)), (np.float64(679.0), np.float64(441.0)),
              (np.float64(684.0), np.float64(441.0)), (np.float64(689.0), np.float64(442.0)),
              (np.float64(694.0), np.float64(443.0)), (np.float64(698.0), np.float64(447.0)),
              (np.float64(699.0), np.float64(452.0)), (np.float64(699.0), np.float64(457.0)),
              (np.float64(698.0), np.float64(462.0)), (np.float64(697.0), np.float64(467.0)),
              (np.float64(696.0), np.float64(472.0)), (np.float64(695.0), np.float64(477.0)),
              (np.float64(695.0), np.float64(482.0)), (np.float64(695.0), np.float64(487.0)),
              (np.float64(696.0), np.float64(492.0)), (np.float64(698.0), np.float64(497.0)),
              (np.float64(702.0), np.float64(502.0)), (np.float64(707.0), np.float64(506.0)),
              (np.float64(712.0), np.float64(508.0)), (np.float64(717.0), np.float64(510.0)),
              (np.float64(722.0), np.float64(511.0)), (np.float64(727.0), np.float64(511.0)),
              (np.float64(732.0), np.float64(511.0)), (np.float64(737.0), np.float64(510.0)),
              (np.float64(742.0), np.float64(510.0)), (np.float64(747.0), np.float64(507.0)),
              (np.float64(752.0), np.float64(502.0)), (np.float64(755.0), np.float64(497.0)),
              (np.float64(758.0), np.float64(492.0)), (np.float64(760.0), np.float64(487.0)),
              (np.float64(761.0), np.float64(482.0)), (np.float64(761.0), np.float64(477.0)),
              (np.float64(761.0), np.float64(472.0)), (np.float64(760.0), np.float64(467.0)),
              (np.float64(759.0), np.float64(462.0)), (np.float64(757.0), np.float64(457.0)),
              (np.float64(757.0), np.float64(452.0)), (np.float64(759.0), np.float64(447.0)),
              (np.float64(764.0), np.float64(443.0)), (np.float64(769.0), np.float64(442.0)),
              (np.float64(774.0), np.float64(442.0)), (np.float64(779.0), np.float64(442.0)),
              (np.float64(784.0), np.float64(442.0)), (np.float64(789.0), np.float64(442.0)),
              (np.float64(794.0), np.float64(442.0)), (np.float64(799.0), np.float64(442.0)),
              (np.float64(804.0), np.float64(442.0)), (np.float64(809.0), np.float64(442.0)),
              (np.float64(814.0), np.float64(442.0)), (np.float64(819.0), np.float64(442.0)),
              (np.float64(824.0), np.float64(442.0)), (np.float64(829.0), np.float64(443.0)),
              (np.float64(834.0), np.float64(443.0)), (np.float64(839.0), np.float64(443.0)),
              (np.float64(844.0), np.float64(443.0)), (np.float64(849.0), np.float64(443.0)),
              (np.float64(854.0), np.float64(442.0))]
    # === Matching side test ===
    bulge_out = Side(points, 1)
    bulge_out_norm = rdp(normalized(bulge_out), epsilon=1)
    bulge_out = Side(bulge_out_norm, 1)
    print("Angle:", bulge_out.angle)
    print("Type:", bulge_out.type)
    visualize_single_side(bulge_out, title="Bulge Out Side")
