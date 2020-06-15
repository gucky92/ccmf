import os
import pickle
import re
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk

from ccmf.circuit import Sign
from .gui_circuit import GUICircuit


class CCMFGUIMixin:
    title = "CCMF"
    width = 720
    height = 480
    n_columns = 4
    bg = "white"
    file_extension = 'circuit'

    def __init__(self):
        self._filename = None
        self._gui_circuit = GUICircuit(self)
        self._root = tk.Tk()
        self._set_title()
        self._init_menu()
        self._canvas = self._init_canvas()
        self._ent_cell_id, self._cell_id_var, self._combo_sign, self._sign_var = self._init_input_widgets()
        self._init_output_widgets()
        self._root.resizable(False, False)
        self._root.lift()
        self._root.attributes('-topmost', True)
        self._root.after_idle(self._root.attributes, '-topmost', False)
        self._root.mainloop()

    def _set_title(self):
        filename = os.path.split(self._filename)[-1] if self._filename else "Untitled"
        self._root.title(f'{filename} - {self.title}')

    def _init_canvas(self):
        canvas = tk.Canvas(self._root, width=self.width, height=self.height, bg=self.bg)
        canvas.bind('<Double-Button-1>', self._handle_add_cell)
        canvas.grid(columnspan=self.n_columns)

        return canvas

    def _init_input_widgets(self):
        column = self.n_columns - 1
        sticky = tk.W + tk.E

        cell_id = tk.StringVar()
        ent_cell_id = tk.Entry(self._root, textvariable=cell_id)
        ent_cell_id.grid(row=1, column=column, sticky=sticky)
        ent_cell_id.bind("<Return>", self._handle_add_cell)

        sign = tk.StringVar()

        combo_sign = ttk.Combobox(self._root, state="readonly", values=[str(i) for i in Sign], textvariable=sign)
        combo_sign.grid(row=2, column=column, sticky=sticky)
        combo_sign.current(0)

        return ent_cell_id, cell_id, combo_sign, sign

    def _init_menu(self):
        menu = tk.Menu(self._root)

        menu_file = tk.Menu(menu, tearoff=0)
        menu_file.add_command(label="New", command=self._handle_new)
        menu_file.add_command(label="Open...", command=self._handle_open)
        menu_file.add_command(label="Save", command=self._handle_save)
        menu_file.add_command(label="Save As...", command=self._handle_save_as)
        # menu_file.add_command(label="Import Data", command=self._handle_import)
        menu_file.add_separator()
        menu_file.add_command(label="Exit", command=self._handle_exit)
        menu.add_cascade(label="File", menu=menu_file)

        menu_edit = tk.Menu(menu, tearoff=0)
        menu_edit.add_command(label="Cut", command=None)
        menu_edit.add_command(label="Copy", command=None)
        menu_edit.add_command(label="Paste", command=None)
        menu.add_cascade(label="Edit", menu=menu_edit)

        menu_run = tk.Menu(menu, tearoff=0)
        menu_run.add_command(label="MAP Estimation", command=self._handle_map_estimation)
        menu_run.add_command(label="MCMC Sampling", command=self._handle_mcmc_sampling)
        menu.add_cascade(label="Run", menu=menu_run)

        menu_help = tk.Menu(menu, tearoff=0)
        menu_help.add_command(label="About", command=None)
        menu.add_cascade(label="Help", menu=menu_help)

        self._root.config(menu=menu)

    def _init_output_widgets(self):
        column = self.n_columns - 2
        sticky = tk.E

        lbl_cell_id = tk.Label(self._root, text="Cell ID: ")
        lbl_cell_id.grid(row=1, column=column, sticky=sticky)

        lbl_sign = tk.Label(self._root, text="Sign: ")
        lbl_sign.grid(row=2, column=column, sticky=sticky)

    @property
    def current_sign(self):
        return Sign[self._sign_var.get().upper()]

    @property
    def canvas(self):
        return self._canvas

    @property
    def circuit(self):
        return self._gui_circuit

    def _read_cell_id(self):
        cell_id = self._cell_id_var.get()
        numeric_suffix = re.sub('.*?([0-9]*)$', r'\1', cell_id)
        try:
            next_cell_id = cell_id[:-len(numeric_suffix)] + str(int(numeric_suffix) + 1)
            self._cell_id_var.set(next_cell_id)
            self._ent_cell_id.icursor(tk.END)
        except ValueError:
            self._cell_id_var.set('')
        return cell_id

    def _handle_add_cell(self, event):
        center = (event.x, event.y) if event.type == tk.EventType.ButtonPress else None
        self._gui_circuit.add_node(self._read_cell_id(), center=center, gui=self)

    def _handle_import(self):
        pass

    def _handle_map_estimation(self):
        pass

    def _handle_mcmc_sampling(self):
        pass

    def _handle_open(self):
        filename = filedialog.askopenfilename(filetypes=[(f"{self.file_extension}", f"*.{self.file_extension}")])
        if filename:
            self._filename = filename
            self._set_title()
            self._handle_new()
            self._gui_circuit.load(pickle.load(open(filename, "rb")))

    def _handle_save(self):
        if self._filename:
            return pickle.dump(self._gui_circuit.save(), open(self._filename, "wb"))
        self._handle_save_as()

    def _handle_save_as(self):
        filename = filedialog.asksaveasfilename(filetypes=[(f"{self.file_extension}", f"*.{self.file_extension}")],
                                                defaultextension=f".{self.file_extension}")
        if filename:
            pickle.dump(self._gui_circuit.save(), open(filename, "wb"))
            self._filename = filename
            self._set_title()

    def _handle_new(self):
        self._gui_circuit.delete_tk()
        self._gui_circuit = GUICircuit(self)

    def _handle_exit(self):
        self._root.quit()
