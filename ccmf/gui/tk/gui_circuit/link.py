import tkinter as tk

import numpy as np

from ccmf.circuit import Sign
from .gui_element import GUIElement


class Link(GUIElement):
    angle = .2
    width = 3

    activefill = {
        Sign.EXCITATORY: 'SeaGreen1',
        Sign.INHIBITORY: 'HotPink1',
        Sign.UNSPECIFIED: 'snow4'
    }

    fill = {
        Sign.EXCITATORY: 'green',
        Sign.INHIBITORY: 'red',
        Sign.UNSPECIFIED: 'black'
    }

    def __init__(self, start_cell, end_cell, gui_circuit):
        self._start_cell = start_cell
        self._end_cell = end_cell
        self._gui_circuit = gui_circuit

        self._line_id = gui_circuit.canvas.create_line(self._coords, arrow=tk.LAST, width=self.width,
                                                       activefill=self.activefill[self._sign],
                                                       fill=self.fill[self._sign])

        super().__init__(gui_circuit, [self._line_id])
        self._lower()

    def _init_binding(self):
        super()._init_binding()

    def _init_menu(self):
        menu = tk.Menu(self._canvas, tearoff=0)
        menu.add_command(label="Show PMF", command=self._handle_show_pmf)
        menu.add_separator()
        menu.add_command(label="Delete", command=self._handle_delete)
        return menu

    @property
    def start(self):
        return self._gui_circuit.nodes[self._start_cell]['node']

    @property
    def end(self):
        return self._gui_circuit.nodes[self._end_cell]['node']

    @property
    def _sign(self):
        try:
            return self._gui_circuit.edges[str(self.start), str(self.end)]['sign']
        except KeyError:
            return self._gui_circuit.gui.current_sign

    @property
    def _coords(self):
        p1 = np.array(self.start.center)
        p2 = np.array(self.end.center)
        dp = p2 - p1
        norm = np.linalg.norm(dp)
        dp = dp / norm if norm else 0
        has_reverse_link = self._gui_circuit.has_edge(self.end, self.start)
        x1, y1 = self._rotate(p1, p1 + dp * self.start.radius, self.angle * has_reverse_link)
        x2, y2 = self._rotate(p2, p2 - dp * self.end.radius, -self.angle * has_reverse_link)
        return x1, y1, x2, y2

    def _drag(self, x, y):
        self.start.drag(x, y)
        self.end.drag(x, y)

    def refresh(self):
        return self._refresh()

    def _refresh(self):
        self._canvas.coords(self._line_id, self._coords)

    def _handle_delete(self):
        self._gui_circuit.remove_edge(self.start, self.end)
        self.delete_tk()
        self._gui_circuit.nodes(), self._gui_circuit.edges()

    def _handle_drag_start(self, event):
        self.start.handle_drag_start(event)
        self.end.handle_drag_start(event)

    def _handle_drag_end(self, event):
        self.start.handle_drag_end(event)
        self.end.handle_drag_end(event)

    def _handle_menu(self, event):
        super()._handle_menu(event)

    def _handle_show_pmf(self):
        pass

    @staticmethod
    def _rotate(origin, point, angle):
        ox, oy = origin
        px, py = point

        qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
        qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
        return qx, qy


class PseudoLink(Link):
    class PseudoNode:
        def __init__(self, center):
            self._center = center

        @property
        def radius(self):
            return 0

        @property
        def center(self):
            return self._center

        @center.setter
        def center(self, center):
            self._center = center

    def __init__(self, start, gui_circuit, end):
        super().__init__(start, self.PseudoNode(end), gui_circuit)

    @property
    def start(self):
        return self._start_cell

    @property
    def end(self):
        return self._end_cell

    def set_end(self, x, y):
        self.end.center = x, y
        self._refresh()

    def handle_delete(self):
        return self._handle_delete()

    def _handle_delete(self):
        self.delete_tk()
