import string
import numpy as np
from enum import Enum
from scipy.spatial.transform import Rotation
import pandas as pd

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
        if isinstance(key, PlayerJoint):
            self.coords_dict[(key.player, key.joint.value)] = value
        elif isinstance(key, tuple) and len(key) == 2:
            self.coords_dict[key] = value
        else:
            raise ValueError("Invalid key type. Expected PlayerJoint or tuple.")

    def __getitem__(self, key):
        if isinstance(key, PlayerJoint):
            return self.coords_dict[(key.player, key.joint.value)]
        elif isinstance(key, tuple) and len(key) == 2:
            return self.coords_dict[key]
        else:
            raise ValueError("Invalid key type. Expected PlayerJoint or tuple.")

    def items(self):
        return self.coords_dict.items()

    def values(self):
        return self.coords_dict.values()

    def keys(self):
        return self.coords_dict.keys()

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
        return d * 4 - 2  # Scale to range -2 to 2

    p = Position()
    for j in PLAYER_JOINTS:
        x = g()
        y = g()
        z = g()
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
    mirrored = {}
    for (player, joint), v in pos.coords_dict.items():
        mirrored_joint = mirror_joint(joint)
        mirrored[(player, mirrored_joint)] = np.array([-v[0], v[1], v[2]])
    return Position(mirrored)

def basically_same(pos1, pos2, tolerance=0.05):
    return all(np.sum((pos1[k] - pos2[k]) ** 2) < tolerance**2 for k in pos1.coords_dict.keys())

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

# New functions for normalization and canonicalization
def normalize_position(pos):
    center = np.mean([v for _,v in pos.coords_dict.items()], axis=0)
    return Position({k: v - center for k, v in pos.coords_dict.items()})

def canonical_reorientation(pos):
    normalized_pos = normalize_position(pos)
    player0_head = normalized_pos[(0, Joint.Head.value)]
    player1_head = normalized_pos[(1, Joint.Head.value)]
    angle = np.arctan2(player1_head[2] - player0_head[2], player1_head[0] - player0_head[0])
    return Reorientation(np.zeros(3), -angle)

def orient_canonically(pos):
    reorientation = canonical_reorientation(pos)
    return apply(reorientation, normalize_position(pos))

# Limb lengths for the spring function
LIMB_LENGTHS = {
    (Joint.LeftAnkle, Joint.LeftKnee): 0.4,
    (Joint.RightAnkle, Joint.RightKnee): 0.4,
    (Joint.LeftKnee, Joint.LeftHip): 0.4,
    (Joint.RightKnee, Joint.RightHip): 0.4,
    (Joint.LeftHip, Joint.Core): 0.2,
    (Joint.RightHip, Joint.Core): 0.2,
    (Joint.Core, Joint.Neck): 0.4,
    (Joint.Neck, Joint.Head): 0.1,
    (Joint.LeftShoulder, Joint.LeftElbow): 0.3,
    (Joint.RightShoulder, Joint.RightElbow): 0.3,
    (Joint.LeftElbow, Joint.LeftWrist): 0.3,
    (Joint.RightElbow, Joint.RightWrist): 0.3,
}

def spring(pos, fixed_joint=None):
    def adjust_joint(joint, force):
        if joint != fixed_joint:
            pos[joint] = pos[joint] + force

    for (j1, j2), length in LIMB_LENGTHS.items():
        for player in range(2):
            p1, p2 = pos[(player, j1.value)], pos[(player, j2.value)]
            direction = p2 - p1
            current_length = np.linalg.norm(direction)
            if current_length > 0:
                force = (current_length - length) * direction / current_length
                adjust_joint((player, j1.value), force * 0.5)
                adjust_joint((player, j2.value), -force * 0.5)

    # Apply y-axis limits
    for k, v in pos.coords_dict.items():
        pos[k] = np.clip(v, [-2, 0, -2], [2, 2, 2])

    return pos

def improved_positions_are_equivalent(pos1, pos2, tolerance=0.05):
    pos1 = orient_canonically(pos1)
    pos2 = orient_canonically(pos2)

    # Check normal orientation
    if basically_same(pos1, pos2, tolerance):
        return True

    # Check mirrored orientation
    if basically_same(pos1, mirror(pos2), tolerance):
        return True

    # Check swapped players
    pos2_swapped = Position({(1 - k[0], k[1]): v for k, v in pos2.coords_dict.items()})
    if basically_same(pos1, pos2_swapped, tolerance):
        return True

    # Check swapped players and mirrored
    if basically_same(pos1, mirror(pos2_swapped), tolerance):
        return True

    return False