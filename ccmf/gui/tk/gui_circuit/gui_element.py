from abc import ABC, abstractmethod

import numpy as np


class GUIElement(ABC):
    def __init__(self, gui_circuit, object_ids):
        self._gui_circuit = gui_circuit
        self._object_ids = object_ids
        self._init_binding()
        self._menu = self._init_menu()

    def __contains__(self, item):
        return item in self._object_ids

    @property
    def _canvas(self):
        return self._gui_circuit.canvas

    @abstractmethod
    def _init_binding(self):
        self._bind("<ButtonPress-1>", self._handle_drag_start)
        self._bind("<B1-Motion>", self._handle_drag_motion)
        self._bind("<ButtonRelease-1>", self._handle_drag_end)
        self._bind("<ButtonPress-3>", self._handle_menu)

    @abstractmethod
    def _init_menu(self):
        pass

    @property
    def _center(self):
        x1, y1, x2, y2 = self._coords
        return (x1 + x2) // 2, (y1 + y2) // 2

    @_center.setter
    def _center(self, center):
        self._move(*tuple(np.subtract(center, self._center)))
        self._refresh()

    @property
    @abstractmethod
    def _coords(self):
        pass

    def _bind(self, *args):
        for object_id in self._object_ids:
            self._canvas.tag_bind(object_id, *args)

    @abstractmethod
    def _drag(self, x, y):
        pass

    def _lower(self):
        for object_id in self._object_ids:
            self._canvas.tag_lower(object_id)

    def _move(self, dx, dy):
        for object_id in self._object_ids:
            self._canvas.move(object_id, dx, dy)

    @abstractmethod
    def _refresh(self):
        pass

    @abstractmethod
    def _handle_delete(self):
        for object_id in self._object_ids:
            self._canvas.delete(object_id)

    @abstractmethod
    def _handle_drag_start(self, event):
        pass

    def _handle_drag_motion(self, event):
        self._drag(event.x, event.y)

    @abstractmethod
    def _handle_drag_end(self, event):
        pass

    @abstractmethod
    def _handle_menu(self, event):
        self._menu.tk_popup(event.x_root, event.y_root)
        self._menu.grab_release()
