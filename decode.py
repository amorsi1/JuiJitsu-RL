import string
from enum import Enum
import numpy as np

BASE62_DIGITS = string.ascii_lowercase + string.ascii_uppercase + string.digits

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
    def __init__(self):
        self.data = {}

    def __setitem__(self, key, value):
        self.data[(key.player, key.joint.value)] = value

    def __getitem__(self, key):
        return self.data[(key.player, key.joint.value)]

    def items(self):
        return self.data.items()


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

def main():
    # Example usage on 'far knee finish' :
    encoded_position = "Kjbf9jYHaX2kJ7eF7XU5aM2TJUdm7mVWbJ2jIWaY0WUlhJZgKlhI1UNPiM1uHeoqVLMdnMT5EhkWWALBjmSCGCiKTOHUh7UzHlhKTLGGhTUxHUg1UBF0hSVFKTk8ZdJHo7UuJMqySdLRaERbIpaFZoIpaKPIGraEWdJkbEQjG2bJW2KSioQgHoiBVVHoogPlFuooSzLwvxQZHTwQUgNwrtP0E5tgTjM7o5TtHoqaVnNqpIUzHZqzWvNipcVJIHpDW6GusfQKKqw4S7MMxeUq"  # Your 276-character encoded position string goes here
    decoded_position = decode_position(encoded_position)

    # To print out the decoded positions:
    for key, value in decoded_position.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()
