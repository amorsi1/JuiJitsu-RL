import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from scipy.spatial.transform import Rotation
import string
from enum import Enum



grapplemap = pd.read_csv('grapplemap_df.csv',dtype={'trans_start_node': 'string', 'trans_end_node': 'string'})
positions = grapplemap[grapplemap['is_position'] == 1]
transitions = grapplemap[grapplemap['is_transition'] == 1]

BASE62_DIGITS = string.ascii_lowercase + string.ascii_uppercase + string.digits

# Decoding logic
def from_base62(c):
    if 'a' <= c <= 'z':
        return ord(c) - ord('a')
    if 'A' <= c <= 'Z':
        return ord(c) - ord('A') + 26
    if '0' <= c <= '9':
        return ord(c) - ord('0') + 52
    raise ValueError(f"Not a base 62 digit: {c}")


class Joint(Enum):
    LeftToe = 0
    RightToe = 1
    LeftHeel = 2
    RightHeel = 3
    LeftAnkle = 4
    RightAnkle = 5
    LeftKnee = 6
    RightKnee = 7
    LeftHip = 8
    RightHip = 9
    LeftShoulder = 10
    RightShoulder = 11
    LeftElbow = 12
    RightElbow = 13
    LeftWrist = 14
    RightWrist = 15
    LeftHand = 16
    RightHand = 17
    LeftFingers = 18
    RightFingers = 19
    Core = 20
    Neck = 21
    Head = 22


JOINT_COUNT = len(Joint)
ENCODED_POS_SIZE = 2 * JOINT_COUNT * 3 * 2


class PlayerJoint:
    def __init__(self, player, joint):
        self.player = player
        self.joint = joint


class Position:
    def __init__(self, input=None):
        self.coords_dict = {}
        if isinstance(input, str):
            self.code = input
            self.coords_dict = decode_position(input)
        elif isinstance(input, dict):
            self.code = None
            self.coords_dict = input
    def __setitem__(self, key, value):
        self.coords_dict[(key.player, key.joint.value)] = value
    def __getitem__(self, key):
        return self.coords_dict[key]
    def items(self):
        return self.coords_dict.items()

def make_player_joints():
    return [PlayerJoint(player, joint) for player in range(2) for joint in Joint]
PLAYER_JOINTS = make_player_joints()


def decode_position(s):
    if len(s) != ENCODED_POS_SIZE:
        raise ValueError(f"Expected string of length {ENCODED_POS_SIZE}, got {len(s)}")

    offset = 0

    def next_digit():
        nonlocal offset
        while offset < len(s) and s[offset].isspace():
            offset += 1
        if offset >= len(s):
            raise ValueError("Unexpected end of string")
        digit = from_base62(s[offset])
        offset += 1
        return digit

    def g():
        d0 = next_digit() * 62
        d = (d0 + next_digit()) / 1000
        return d

    p = Position()
    for j in PLAYER_JOINTS:
        x = g() - 2
        y = g()
        z = g() - 2
        p[j] = np.array([x, y, z])

    return p


# Position comparison logic
class Reorientation:
    def __init__(self, offset, angle):
        self.offset = offset
        self.angle = angle


class PositionReorientation:
    def __init__(self, reorientation, swap_players, mirror):
        self.reorientation = reorientation
        self.swap_players = swap_players
        self.mirror = mirror



# Create a mapping of joint names to their mirror counterparts
MIRROR_JOINTS = {
    Joint.LeftToe: Joint.RightToe,
    Joint.LeftHeel: Joint.RightHeel,
    Joint.LeftAnkle: Joint.RightAnkle,
    Joint.LeftKnee: Joint.RightKnee,
    Joint.LeftHip: Joint.RightHip,
    Joint.LeftShoulder: Joint.RightShoulder,
    Joint.LeftElbow: Joint.RightElbow,
    Joint.LeftWrist: Joint.RightWrist,
    Joint.LeftHand: Joint.RightHand,
    Joint.LeftFingers: Joint.RightFingers,
}
# Add reverse mappings
MIRROR_JOINTS.update({v: k for k, v in MIRROR_JOINTS.items()})

def mirror_joint(joint: int) -> int:
    joint_enum = Joint(joint)
    if joint_enum in MIRROR_JOINTS:
        return MIRROR_JOINTS[joint_enum].value
    return joint

def xz(v):
    return np.array([v[0], v[2]])


def angle(v):
    return np.arctan2(v[1], v[0])


def yrot(angle):
    return Rotation.from_rotvec([0, angle, 0]).as_matrix()


def apply(reorientation, pos):
    r = Rotation.from_rotvec([0, reorientation.angle, 0])
    return Position({k: r.apply(v) + reorientation.offset for k, v in pos.coords_dict.items()})


def mirror(pos):
    return Position({k: np.array([-v[0], v[1], v[2]]) for k, v in pos.coords_dict.items()})


def basically_same(pos1, pos2, tolerance=0.05):
    return all(np.sum((pos1[k] - pos2[k]) ** 2) < tolerance for k in pos1.coords_dict.keys())


def is_reoriented_without_mirror_and_swap(a, b):
    a0h, a1h = a[(0, Joint.Head.value)], a[(1, Joint.Head.value)]
    b0h, b1h = b[(0, Joint.Head.value)], b[(1, Joint.Head.value)]
    angle_off = angle(xz(b1h - b0h)) - angle(xz(a1h - a0h))
    r = Reorientation(b0h - yrot(angle_off) @ np.append(a0h, 1)[:3], angle_off)
    if basically_same(apply(r, a), b):
        return r
    return None


def is_reoriented_without_swap(a, b):
    r = is_reoriented_without_mirror_and_swap(a, b)
    if r:
        return PositionReorientation(r, False, False)

    r = is_reoriented_without_mirror_and_swap(a, mirror(b))
    if r:
        return PositionReorientation(r, False, True)

    return None


def is_reoriented(a, b):
    def head2head(p):
        player1_headpos = p[(0, Joint.Head.value)]
        player2_headpos = p[(1, Joint.Head.value)]
        return np.sum((player1_headpos - player2_headpos) ** 2)

    if abs(head2head(a) - head2head(b)) > 0.05:
        return None

    r = is_reoriented_without_swap(a, b)
    if not r:
        b_swapped = Position({(1 - k[0], k[1]): v for k, v in b.coords_dict.items()})
        r = is_reoriented_without_swap(a, b_swapped)
        if r:
            r.swap_players = True

    return r


def positions_are_equivalent(pos1, pos2):
    return is_reoriented(pos1, pos2) is not None

def main():
    honey = transitions[transitions['description'] == 'to honey']['start_position'].iloc[1]
    return positions_are_equivalent(Position(honey), Position(honey))

if __name__ == "__main__":
    main()
