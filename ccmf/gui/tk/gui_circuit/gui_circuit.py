import networkx as nx

from ccmf.circuit import Circuit
from .link import Link
from .node import Node


class GUICircuit(Circuit):
    def __init__(self, gui, **attr):
        self._gui = gui
        super().__init__(**attr)

    def load(self, saved_circuit):
        transform = {}
        for key, value in saved_circuit.nodes.items():
            transform[key] = self.add_node(key, center=value['center'])
        for key, value in saved_circuit.edges.items():
            self.add_edge(transform[key[0]], transform[key[1]], sign=value['sign'])
        for key, value in self.nodes.items():
            value['node'].refresh()

    def save(self):
        circuit = Circuit()
        for key, value in self.nodes.items():
            circuit.add_node(key, center=value['node'].center)
        for key, value in self.edges.items():
            circuit.add_edge(*key, sign=value['sign'])
        return circuit

    def delete_tk(self):
        for i, value in self.nodes.items():
            value['node'].delete_tk()
        for i, value in self.edges.items():
            value['link'].delete_tk()

    @property
    def gui(self):
        return self._gui

    @property
    def canvas(self):
        return self.gui.canvas

    def add_node(self, node_for_adding, **attr):
        cell_id = self._get_unique_id(node_for_adding)
        super().add_node(cell_id, node=Node(cell_id, self, attr['center'] if 'center' in attr else None))
        return cell_id

    def remove_node(self, n):
        if isinstance(n, Node):
            return super().remove_node(str(n))
        return super().remove_node(n)

    def has_edge(self, u, v):
        return super().has_edge(str(u), str(v))

    def add_edge(self, u, v, **attr):
        sign = attr['sign'] if 'sign' in attr else self.gui.current_sign
        super().add_edge(str(u), str(v), sign=sign)
        link = Link(str(u), str(v), self)
        nx.set_edge_attributes(self, {(str(u), str(v)): link}, 'link')

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
