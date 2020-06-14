import networkx as nx


class Circuit(nx.DiGraph):
    def __init__(self, **attr):
        super().__init__(**attr)
