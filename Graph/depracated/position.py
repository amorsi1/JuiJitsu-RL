import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import pandas as pd
from scipy.spatial import procrustes

grapplemap = pd.read_csv('files/grapplemap_df.csv', dtype={'trans_start_node': 'string', 'trans_end_node': 'string'})
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

class Position:
    def __init__(self, input_data):
        if isinstance(input_data, str) and len(input_data) == 276:
            self.coords = self.decode_position(input_data)
            self.codeblock = input_data
        elif isinstance(input_data, dict) and len(input_data) == 46:
            self.coords = input_data
        elif isinstance(input_data, list) and np.shape(input_data) == (46, 3):
            #Assuming this list was generated from a Position.coords dict, and thus the ordering is perserved. If not, then the generated position will be entirely wrong
            coords = {}
            for player in range(2):
                for joint in Joint:
                    coords[(player, joint.value)] = input_data.pop(0)
            self.coords = coords
        else:
            raise ValueError("Input must be either a string, a dictionary, or a list")
        if self.coords:
            self.player0 = {k: v for k, v in self.coords.items() if k[0] == 0}
            self.player1 = {k: v for k, v in self.coords.items() if k[0] == 1}
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

def mirror_joint(joint: int) -> int:
    joint_enum = Joint(joint)
    if joint_enum in MIRROR_JOINTS:
        return MIRROR_JOINTS[joint_enum].value
    return joint
def mirror(pos: Position) -> Position:
    mirrored = {}
    for (player, joint), v in pos.coords.items():
        mirrored_joint = mirror_joint(joint)
        mirrored[(player, mirrored_joint)] = np.array([-v[0], v[1], v[2]])
    return Position(mirrored)

def distance_from_head(pos: Position,player_num: int) -> list:
    assert player_num in (0,1), 'player_num value should be 0 or 1, denoting which player to apply function to'
    players_coords = {k: v for k, v in pos.coords.items() if k[0] == player_num}
    head_coord = pos[(player_num, Joint.Head.value)]
    distances = []
    for coord in players_coords.values():
        distances.append(np.subtract(head_coord, coord))
    return distances
def calc_limb_distances(pos: Position) -> dict:
    player0,player1 = distance_from_head(pos,0),distance_from_head(pos,1)
    return {0: player0, 1: player1}

def same_limb_distances(pos1: Position,pos2: Position, tolerance=0.05) -> bool:
    pos1_distances = calc_limb_distances(pos1)
    pos2_distances = calc_limb_distances(pos2)

    for player in (0, 1):
        distances1 = pos1_distances[player]
        distances2 = pos2_distances[player]
        for dist1,dist2 in zip(distances1,distances2):
            # print(np.sum(dist1),np.sum(dist2))
            if not np.allclose(dist1, dist2, atol=tolerance):
                return False
    return True
def pos_to_list(pos: Position) -> list:
    """returns a list of a positions coordinates"""
    if isinstance(pos,str): pos = Position(pos)
    return list(pos.coords.values())
def procrustes_analysis(pos1: Position, pos2: Position,tolerance=0.05) -> bool:
    """ performs orthogonal procrustes analysis to see if one position can be rotated and reflected into another """
    _,_,disparity = procrustes(pos_to_list(pos1),pos_to_list(pos2))
    if disparity < tolerance:
        return True
    else:
        return False
# def basically_same(pos1, pos2, tolerance=0.12):
#     return all(np.linalg.norm(np.abs(pos1[k]) - np.abs(pos2[k])) < tolerance for k in pos1.coords.keys())



def head2head(p):
    return np.sum((p[(0, Joint.Head.value)] - p[(1, Joint.Head.value)]) ** 2)

def swap_players(pos: Position) -> Position:
    # swapped_dict = {}
    # for (key1, value1), (key2, value2) in zip(pos.player0.items(), pos.player1.items()):
    #     swapped_dict[key1] = value2
    #     swapped_dict[key2] = value1
    # return Position(swapped_dict)
    keys = list(pos.coords.keys())
    vals = list(pos.coords.values())
    swapped_vals = vals[23:] + vals[:23]
    swapped_dict = {key: value for key, value in zip(keys, swapped_vals)}
    return Position(swapped_dict)



def is_reoriented(a, b):
    # quick way to filter out positions that are not at all similar
    if abs(head2head(a) - head2head(b)) > 0.05:
        return None

    r = same_limb_distances(a, b)
    if r is not True:
        # procrustes
        r = procrustes_analysis(a,b)
    if r is not True:
        #swapped players
        b_swapped = swap_players(b)
        r = procrustes_analysis(a, b_swapped)
    if r is not True:
        #mirrored players
        r = procrustes_analysis(a,mirror(b))
    if r is not True:
        #mirrored and swapped
        r = procrustes_analysis(a,mirror(b_swapped))
    return r


# def canonical_reorientation_with_mirror(p):
#     def normal_rotation(pos):
#         p0h, p1h = pos[(0, Joint.Head.value)], pos[(1, Joint.Head.value)]
#         return -angle(xz(p1h - p0h))
#
#     def normal_translation(pos):
#         center = np.mean([v for v in pos.coords.values()], axis=0)
#         return -center
#
#     rotation_angle = normal_rotation(p)
#     offset = normal_translation(apply(Reorientation(np.zeros(3), rotation_angle), p))
#     mirror_flag = apply(Reorientation(offset, rotation_angle), p)[(1, Joint.Head.value)][0] >= 0
#
#     return PositionReorientation(Reorientation(offset, rotation_angle), False, mirror_flag)

def positions_are_equivalent(pos1, pos2):
    return is_reoriented(pos1, pos2)


# grapplemap = pd.read_csv('grapplemap_df.csv',dtype={'trans_start_node': 'string', 'trans_end_node': 'string'})
# positions = grapplemap[grapplemap['is_position'] == 1]
# example_row = positions.iloc[np.random.randint(0, high=len(positions)-1),:]['code']
# example_pos = Position(example_row)
# print('hold')

