import tkinter as tk
from itertools import chain

import numpy as np

from .gui_element import GUIElement
from .link import PseudoLink


class Node(GUIElement):
    """GUI component for cell.

    """
    activefill = "cyan"
    fill = "white"
    radius = 20
    width = 3
    grid = radius

    def __init__(self, cell, gui_circuit, center=None):
        self._cell = cell
        self._gui_circuit = gui_circuit
        self.__center = None

        self._drag_offset = None
        self._pseudo_link = None

        if center is None:
            center = self._canvas.winfo_width() // 2, self._canvas.winfo_height() // 2

        self._oval_id = gui_circuit.canvas.create_oval(*self._bbox(center), width=self.width, fill=self.fill,
                                                       activefill=self.activefill)
        text_id = gui_circuit.canvas.create_text(*center, text=cell, state=tk.DISABLED)

        super().__init__(gui_circuit, [self._oval_id, text_id])

    def __str__(self):
        return str(self._cell)

    def _init_binding(self):
        super()._init_binding()

        self._bind("<ButtonPress-2>", self._handle_link_start)
        self._bind("<B2-Motion>", self._handle_link_motion)
        self._bind("<ButtonRelease-2>", self._handle_link_end)

    def _init_menu(self):
        menu = tk.Menu(self._canvas, tearoff=0)
        menu.add_command(label="Delete", command=self._handle_delete)
        return menu

    @property
    def center(self):
        try:
            self.__center = self._center
        except tk.TclError:
            pass
        return self.__center

    def drag(self, x, y):
        return self._drag(x, y)

    def handle_drag_start(self, event):
        self._handle_drag_start(event)

    def handle_drag_end(self, event):
        self._handle_drag_end(event)

    @property
    def _coords(self):
        return self._canvas.coords(self._oval_id)

    def _bbox(self, center):
        x, y = center
        return x - self.radius, y - self.radius, x + self.radius, y + self.radius

    def _drag(self, x, y):
        self._center = (np.subtract((x, y), self._drag_offset) // self.grid * self.grid).astype(int)

    def refresh(self):
        for i in chain(self._gui_circuit.in_edges(str(self)), self._gui_circuit.out_edges(str(self))):
            self._gui_circuit.edges[i]['link'].refresh()

    def _handle_delete(self):
        for i in chain(self._gui_circuit.in_edges(str(self)), self._gui_circuit.out_edges(str(self))):
            self._gui_circuit.edges[i]['link'].delete_tk()

        self._gui_circuit.remove_node(self)
        self.delete_tk()

    def _handle_drag_start(self, event):
        self._drag_offset = np.subtract((event.x, event.y), self._center)

    def _handle_drag_end(self, event):
        self._drag_offset = None

    def _handle_menu(self, event):
        if self._drag_offset is None and self._pseudo_link is None:
            super()._handle_menu(event)

    def _handle_link_start(self, event):
        self._pseudo_link = PseudoLink(self, self._gui_circuit, (event.x, event.y))

    def _handle_link_motion(self, event):
        self._pseudo_link.set_end(event.x, event.y)

    def _handle_link_end(self, event):
        self._pseudo_link.handle_delete()
        self._pseudo_link = None

        end = self._gui_circuit.get_closest_node(event.x, event.y, self)

        if end and np.linalg.norm(np.subtract(end.center, (event.x, event.y))) < end.radius * 2:
            self._gui_circuit.add_edge(self, end)
            self.refresh()
