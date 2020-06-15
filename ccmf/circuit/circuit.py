import re

import networkx as nx


class Circuit(nx.DiGraph):
    def __init__(self, **attr):
        super().__init__(**attr)

    def add_node(self, node_for_adding, **attr):
        unique_cell = self._get_unique_id(node_for_adding)
        super().add_node(unique_cell, **attr)
        return unique_cell

    def _get_unique_id(self, cell):
        if not cell:
            if self.number_of_nodes():
                return self._get_unique_id(list(self.nodes(1))[-1][0])
            return "1"

        if cell in self:
            try:
                numeric_suffix = re.sub('.*?-([0-9]*)$', r'\1', cell)
                return self._get_unique_id(cell[:-len(numeric_suffix)] + str(int(numeric_suffix) + 1))
            except ValueError:
                return self._get_unique_id(cell + '-1')
        return cell

    @property
    def inputs(self):
        return [cell for cell, in_degrees in self.in_degree() if in_degrees == 0]

    @property
    def outputs(self):
        return [cell for cell, in_degrees in self.in_degree() if in_degrees]
