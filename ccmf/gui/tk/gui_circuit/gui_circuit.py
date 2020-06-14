from ccmf.circuit import Circuit
from .node import Node
from .link import Link


class GUICircuit(Circuit):
    def __init__(self, gui, **attr):
        self._gui = gui
        super().__init__(**attr)

    @property
    def gui(self):
        return self._gui

    @property
    def canvas(self):
        return self.gui.canvas

    def add_node(self, node_for_adding, **attr):
        cell_id = self._get_unique_id(node_for_adding)
        super().add_node(cell_id, node=Node(cell_id, self, attr['center'] if 'center' in attr else None))

    def remove_node(self, n):
        if isinstance(n, Node):
            return super().remove_node(str(n))
        return super().remove_node(n)

    def has_edge(self, u, v):
        return super().has_edge(str(u), str(v))

    def add_edge(self, u, v, **attr):
        return super().add_edge(str(u), str(v), link=Link(str(u), str(v), self))

    def remove_edge(self, u, v):
        return super().remove_edge(str(u), str(v))

    def _get_node_by_object_id(self, object_id):
        for i in self.nodes.values():
            if object_id in i['node']:
                return i['node']

    def get_closest_node(self, x, y, exclusion=None):
        for object_id in self.canvas.find_closest(x, y):
            node = self._get_node_by_object_id(object_id)
            if node and node != exclusion:
                return node
