import hashlib
import os


def writeIndex(filename, g, dbHash):
    with open(filename, 'w') as f:
        f.write(dbHash + '\n')
        for n in seqnums(g):
            f.write(str(g[n].from_node.index) + ' ' + str(g[n].to_node.index) + ' ')


def loadGraph_file(filename):
    with open(filename, 'rb') as ff:
        db = ff.read().decode('utf-8')
        dbhash = hashlib.md5(db.encode('utf-8')).hexdigest()
        indexFile = filename + ".index"
        connections = []
        if os.path.exists(indexFile):
            with open(indexFile, 'r') as f:
                indexHash = f.readline().strip()
                if indexHash == dbhash:
                    for line in f:
                        from_index, to_index = map(int, line.split())
                        connections.append((NodeNum(from_index), NodeNum(to_index)))
        edges = readSeqs(db, len(db))
        pp = []
        def is_pos(s):
            return len(s.positions) == 1
        for e in edges:
            if is_pos(e):
                pp.append(NamedPosition(e.positions[0], e.description, e.line_nr))
        edges = [e for e in edges if not is_pos(e)]
        if connections:
            return Graph(pp, edges, connections)
        g = Graph(pp, edges)
        writeIndex(indexFile, g, dbhash)
        return g

import math
from typing import List, Optional, Set

base62digits = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
assert len(base62digits) == 62 + 1, "hm"

def fromBase62(c):
    if 'a' <= c <= 'z':
        return ord(c) - ord('a')
    if 'A' <= c <= 'Z':
        return ord(c) - ord('A') + 26
    if '0' <= c <= '9':
        return ord(c) - ord('0') + 52
    raise ValueError("not a base 62 digit: " + c)

def desc(n):
    desc = n.description
    return "?" if not desc else desc[0]

encoded_pos_size = 2 * joint_count * 3 * 2 + 4 * 5

def decodePosition(s):
    offset = 0
    def nextdigit():
        nonlocal offset
        while s[offset].isspace():
            offset += 1
        digit = fromBase62(s[offset])
        offset += 1
        return digit
    def g():
        d0 = nextdigit() * 62
        d = (d0 + nextdigit()) / 1000
        return d
    p = Position()
    for j in playerJoints:
        p[j] = (g() - 2, g(), g() - 2)
    return p

def properties_in_desc(desc: List[str]) -> Set[str]:
    r = set()
    decl = "properties:"
    for line in desc:
        if line.startswith(decl):
            props = line[len(decl):].split()
            r.update(props)
    return r

def readSeqs(b, e):
    v = []
    desc = []
    last_was_position = False
    line_nr = 0
    try:
        while b < e:
            is_position = b[0] == ' '
            if is_position:
                if not last_was_position:
                    assert desc
                    props = properties_in_desc(desc)
                    v.append(Sequence(
                        description=desc,
                        positions=[],
                        line_nr=line_nr - len(desc),
                        detailed='detailed' in props,
                        bidirectional='bidirectional' in props
                    ))
                    desc = []
                if len(e - b) < encoded_pos_size:
                    raise RuntimeError("Insufficient data")
                v[-1].positions.append(decodePosition(b))
                b += encoded_pos_size
                line_nr += 4
            else:
                t = b
                while b != e and b[0] != '\n':
                    b += 1
                desc.append(b[t:b])
                line_nr += 1
                if b != e:
                    b += 1
            last_was_position = is_position
    except Exception as ex:
        raise ValueError(f"at line {line_nr}: {str(ex)}") from ex
    return v

def write_position(o, p):
    s = ""
    def g(d):
        nonlocal s
        i = int(round(d * 1000))
        assert 0 <= i < 4000
        s += base62digits[i // 62]
        s += base62digits[i % 62]
    for j in playerJoints:
        g(p[j][0] + 2)
        g(p[j][1])
        g(p[j][2] + 2)
    n = len(s) // 4
    for i in range(4):
        o.write("    " + s[i * n:(i + 1) * n] + '\n')

def write_sequence(o, s):
    for l in s.description:
        o.write(l + '\n')
    for p in s.positions:
        write_position(o, p)





def nodenums(g):
    return range(g.num_nodes())

def seqnums(g):
    return range(g.num_sequences())

def insert(g, sequence):
    # Implementation of inserting a sequence into the graph
    pass

def erase_sequence(g, seqnum):
    # Implementation of erasing a sequence from the graph
    pass

def split_at(g, position_in_sequence):
    # Implementation of splitting a sequence at a given position
    pass

def end(s):
    return len(s.positions)

def last_pos(s):
    return len(s.positions) - 1

def pos_loc(pis, g):
    seq = g[pis.sequence]
    if pis.position == last_pos(seq):
        return Location(SegmentInSequence(pis.sequence, pis.position - 1), 1)
    else:
        return Location(SegmentInSequence(pis.sequence, pis.position), 0)

def last_pos_in(s, g):
    return s * last_pos(g[s])

def posnums(g, s):
    return range(end(g[s]))

def positions(g, s):
    return [s * p for p in posnums(g, s)]

def last_segment(s):
    return len(s.positions) - 2

def last_segment(s, g):
    return s * last_segment(g[s])

def num_segments(s):
    return len(s.positions) - 1

def segments(s):
    return range(num_segments(s))

def end_pos(s):
    return len(s.positions)

def positions(s):
    return range(end_pos(s))

def segments(seq, g):
    return [seq * seg for seg in segments(g[seq])]

# def node_at(g, pis):
#     if pis == first_pos_in(pis.sequence):
#         return g[pis.sequence].from_node
#     if pis == last_pos_in(pis.sequence, g):
#         return g[pis.sequence].to_node
#     return None


from typing import List, Optional

class Sequence:
    def __init__(self, description: List[str], positions: List[Position], line_nr: Optional[int] = None, detailed: bool = False, bidirectional: bool = False):
        self.description = description
        self.positions = positions
        self.line_nr = line_nr
        self.detailed = detailed
        self.bidirectional = bidirectional

    def __getitem__(self, n: PosNum) -> Position:
        return self.positions[n.index]