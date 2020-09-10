import graphviz


class Graph(object):

    def __init__(self, gdict=None, is_directed=True):
        self.is_directed = is_directed
        if gdict is None:
            gdict = {}
        self.gdict = gdict

    def get_nodes(self):
        return list(self.gdict.keys())

    def get_edges(self):
        edgename = []
        for vrtx in self.gdict:
            for nxtvrtx in self.gdict[vrtx]:
                if {nxtvrtx, vrtx} not in edgename:
                    edgename.append((vrtx, nxtvrtx))
        return edgename

    def add_node(self, vrtx):
        if vrtx not in self.gdict:
            self.gdict[vrtx] = []

    def add_edge(self, vrtx1, vrtx2):
        if vrtx1 in self.gdict:
            self.gdict[vrtx1].append(vrtx2)
        else:
            self.gdict[vrtx1] = [vrtx2]

    def get_node(self, node_name):
        for node in self.get_nodes():
            if node.name == node_name:
                return node

    def to_namegraph(self):
        g = Graph(is_directed=self.is_directed)

        for v in self.get_nodes():
            g.add_node(v.name if hasattr(v, 'name') else v)

        for v1, v2 in self.get_edges():
            g.add_edge(v1.name if hasattr(v1, 'name') else v1, v2.name if hasattr(v2, 'name') else v2)

        return g

    def to_graphviz(self):
        if self.is_directed:
            g = graphviz.Digraph()
        else:
            g = graphviz.Graph()

        for vrtx in self.get_nodes():
            g.node(vrtx.name) if hasattr(vrtx, 'name') else vrtx

        for vrtx1, vrtx2 in self.get_edges():
            g.edge(vrtx1.name if hasattr(vrtx1, 'name') else vrtx1, vrtx2.name if hasattr(vrtx2, 'name') else vrtx2)

        return g

    def __repr__(self):
        return str(self.gdict)
