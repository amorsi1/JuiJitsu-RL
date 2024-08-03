from utils import  *



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