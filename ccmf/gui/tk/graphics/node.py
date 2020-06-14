import tkinter as tk

import numpy as np

from .graph_element import GraphElement


class Node(GraphElement):
    activefill = "cyan"
    fill = "white"
    radius = 20
    width = 3

    def __init__(self, cell, canvas, center):
        self._cell = cell
        self._drag_offset = None
        self._pseudo_link = None

        self._oval_id = canvas.create_oval(*self._bbox(center), width=self.width, fill=self.fill,
                                           activefill=self.activefill)
        text_id = canvas.create_text(*center, text=str(cell), state=tk.DISABLED)

        super().__init__(canvas, [self._oval_id, text_id])

    def __str__(self):
        return str(self._cell)

    def _init_binding(self):
        super()._init_binding()

        self._bind("<ButtonPress-2>", self._handle_edge_start)
        self._bind("<B2-Motion>", self._handle_edge_motion)
        self._bind("<ButtonRelease-2>", self._handle_edge_end)

    def _init_menu(self):
        menu = tk.Menu(self._canvas, tearoff=0)
        menu.add_command(label="Delete", command=None)
        return menu

    @property
    def center(self):
        return self._center

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
        self._center = np.subtract((x, y), self._drag_offset)

    def _refresh(self):
        pass

    def _handle_drag_start(self, event):
        self._drag_offset = np.subtract((event.x, event.y), self._center)

    def _handle_drag_end(self, event):
        self._drag_offset = None

    def _handle_menu(self, event):
        if self._drag_offset is None and self._pseudo_link is None:
            super()._handle_menu(event)

    def _handle_edge_start(self, event):
        pass
        # self._pseudo_link = PseudoLink(self, self._graph)

    def _handle_edge_motion(self, event):
        pass
        # self._pseudo_link.set_end(event.x, event.y)

    def _handle_edge_end(self, event):
        pass
        # self._pseudo_link.__del__()
        # self._pseudo_link = None
        # end_node = self._graph.get_closest_node(event.x, event.y, exclusion=self)
        # if end_node and np.linalg.norm(np.subtract(end_node.center, (event.x, event.y))) < end_node.radius * 2:
        #     self._graph.create_link(self, end_node)
        #     self._refresh()
