import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import pandas as pd

grapplemap = pd.read_csv('grapplemap_df.csv',dtype={'trans_start_node': 'string', 'trans_end_node': 'string'})
positions = grapplemap[grapplemap['is_position'] == 1]
transitions = grapplemap[grapplemap['is_transition'] == 1]

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


@dataclass
class Reorientation:
    offset: np.ndarray
    angle: float


@dataclass
class PositionReorientation:
    reorientation: Reorientation
    swap_players: bool
    mirror: bool


class Position:
    def __init__(self, input_data):
        if isinstance(input_data, str):
            self.coords = self.decode_position(input_data)
        elif isinstance(input_data, dict):
            self.coords = input_data
        else:
            raise ValueError("Input must be either a string or a dictionary")

    def decode_position(self, s):
        if len(s) != ENCODED_POS_SIZE:
            raise ValueError(f"Expected string of length {ENCODED_POS_SIZE}, got {len(s)}")

        def from_base62(c):
            if 'a' <= c <= 'z':
                return ord(c) - ord('a')
            if 'A' <= c <= 'Z':
                return ord(c) - ord('A') + 26
            if '0' <= c <= '9':
                return ord(c) - ord('0') + 52
            raise ValueError(f"Not a base 62 digit: {c}")

        coords = {}
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

        for player in range(2):
            for joint in Joint:
                x, y, z = g(), g(), g()
                coords[(player, joint.value)] = np.array([x, y, z])

        return coords

    def __getitem__(self, key):
        return self.coords[key]

    def __setitem__(self, key, value):
        self.coords[key] = value

    def items(self):
        return self.coords.items()


def xz(v):
    return np.array([v[0], v[2]])


def angle(v):
    return np.arctan2(v[1], v[0])


def yrot(theta):
    return np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])


def apply(reorientation, pos):
    r = yrot(reorientation.angle)
    return Position({k: r @ v + reorientation.offset for k, v in pos.coords.items()})


def mirror(pos):
    return Position({k: np.array([-v[0], v[1], v[2]]) for k, v in pos.coords.items()})


def basically_same(pos1, pos2, tolerance=0.05):
    return all(np.linalg.norm(np.abs(pos1[k]) - np.abs(pos2[k])) < tolerance for k in pos1.coords.keys())


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


def head2head(p):
    return np.sum((p[(0, Joint.Head.value)] - p[(1, Joint.Head.value)]) ** 2)


def is_reoriented(a, b):
    if abs(head2head(a) - head2head(b)) > 0.05:
        return None

    r = is_reoriented_without_swap(a, b)

    if not r:
        b_swapped = Position({(1 - k[0], k[1]): v for k, v in b.coords.items()})
        r = is_reoriented_without_swap(a, b_swapped)
        if r:
            r.swap_players = True

    return r


def canonical_reorientation_with_mirror(p):
    def normal_rotation(pos):
        p0h, p1h = pos[(0, Joint.Head.value)], pos[(1, Joint.Head.value)]
        return -angle(xz(p1h - p0h))

    def normal_translation(pos):
        center = np.mean([v for v in pos.coords.values()], axis=0)
        return -center

    rotation_angle = normal_rotation(p)
    offset = normal_translation(apply(Reorientation(np.zeros(3), rotation_angle), p))
    mirror_flag = apply(Reorientation(offset, rotation_angle), p)[(1, Joint.Head.value)][0] >= 0

    return PositionReorientation(Reorientation(offset, rotation_angle), False, mirror_flag)


def orient_canonically_with_mirror(p):
    return apply(canonical_reorientation_with_mirror(p).reorientation, p)


def positions_are_equivalent(pos1, pos2):
    return is_reoriented(pos1, pos2) is not None


# Usage example
# def test_positions_are_equivalent(encoded_pos1, encoded_pos2):
#     pos1 = Position(encoded_pos1)
#     pos2 = Position(encoded_pos2)
#     return positions_are_equivalent(pos1, pos2)
