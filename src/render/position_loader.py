"""
Loads 3D position and transition frame data from GrappleMap JSON files
into lookup dicts, separate from the NetworkX graph.
"""
import json
import os
from typing import Dict, List
from dotenv import load_dotenv

# Path to the GrappleMap data files
load_dotenv(".env.template") # default .env template, safe to commit to repo
load_dotenv(".env", override=True) # override if a private .env file is provided

DATA_DIR = os.environ.get('GRAPH_FILES_DIR', 'GrappleMap_files/')


def _parse_joint(j: dict) -> List[float]:
    """Convert {_x, _y, _z} dict to [x, y, z] list."""
    return [j['_x'], j['_y'], j['_z']]


def _parse_frame(frame: list) -> List[List[List[float]]]:
    """Parse a frame: [[player0_joints], [player1_joints]] where each joint is [x,y,z]."""
    return [[_parse_joint(joint) for joint in player] for player in frame]


def _parse_reo(reo: dict) -> dict:
    """Convert GrappleMap reorientation metadata into JSON-friendly primitives."""
    return {
        'mirror': reo['mirror'],
        'swap_players': reo['swap_players'],
        'angle': reo['angle'],
        'offset': _parse_joint(reo['offset']),
    }


def load_positions(nodes_path: str = None) -> Dict[int, List[List[List[float]]]]:
    """
    Load node positions: node_id -> [[player0 joints], [player1 joints]]
    Each joint is [x, y, z]. Each player has 23 joints.
    """
    if nodes_path is None:
        nodes_path = os.path.join(DATA_DIR, 'nodes.json')
    with open(nodes_path, 'r') as f:
        nodes = json.load(f)

    positions = {}
    for node in nodes:
        if 'position' in node:
            positions[node['id']] = _parse_frame(node['position'])
    return positions


def load_transition_frames(transitions_path: str = None) -> Dict[int, dict]:
    """
    Load transition data: transition_id -> {'frames': [...], 'detailed': bool, ...}.
    Each frame is [[player0 joints], [player1 joints]], each joint is [x, y, z].
    The 'detailed' flag indicates whether the transition has fine-grained keyframes
    (if False, the viewer should double frames by inserting midpoints).
    """
    if transitions_path is None:
        transitions_path = os.path.join(DATA_DIR, 'transitions.json')
    with open(transitions_path, 'r') as f:
        transitions = json.load(f)

    transition_data = {}
    for t in transitions:
        if 'frames' in t:
            detailed = 'detailed' in t.get('properties', [])
            transition_data[t['id']] = {
                'frames': [_parse_frame(frame) for frame in t['frames']],
                'detailed': detailed,
                'from_node': t['from']['node'],
                'to_node': t['to']['node'],
                'from_reo': _parse_reo(t['from']['reo']),
                'to_reo': _parse_reo(t['to']['reo']),
            }
    return transition_data


def load_raw_transition_index(transitions_path: str | None = None) -> dict[int, dict]:
    """Load transitions.json and index by ID without parsing frame data.

    Returns raw JSON dicts keyed by transition ID. Frame parsing is deferred
    to `parse_transition()` for on-demand use.
    """
    if transitions_path is None:
        transitions_path = os.path.join(DATA_DIR, 'transitions.json')
    with open(transitions_path, 'r') as f:
        transitions = json.load(f)
    return {t['id']: t for t in transitions if 'frames' in t}


def parse_transition(raw: dict) -> dict:
    """Parse a single raw transition dict into the standard format.

    Returns {'frames': [...], 'detailed': bool, 'from_node': int, 'to_node': int,
             'from_reo': dict, 'to_reo': dict}.
    """
    detailed = 'detailed' in raw.get('properties', [])
    return {
        'frames': [_parse_frame(frame) for frame in raw['frames']],
        'detailed': detailed,
        'from_node': raw['from']['node'],
        'to_node': raw['to']['node'],
        'from_reo': _parse_reo(raw['from']['reo']),
        'to_reo': _parse_reo(raw['to']['reo']),
    }


def load_all(nodes_path: str = None, transitions_path: str = None):
    """Load both positions and transition frames."""
    return load_positions(nodes_path), load_transition_frames(transitions_path)
