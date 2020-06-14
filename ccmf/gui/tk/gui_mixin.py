import tkinter as tk
from tkinter import ttk

from ccmf.circuit import Circuit, Sign, Cell
from .graphics import Node


class CCMFGUIMixin:
    title = "CCMF"
    width = 720
    height = 480
    n_columns = 4
    bg = "white"
    file_extension = 'circuit'

    def __init__(self):
        self._circuit = Circuit()
        self._root = tk.Tk()
        self._root.title(self.title)
        self._init_menu()
        self._canvas = self._init_canvas()
        self._cell_id_var, self._combo_connection, self._connection_var = self._init_input_widgets()
        self._init_output_widgets()
        self._root.resizable(False, False)
        self._root.mainloop()

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

        return cell_id, combo_sign, sign

    def _init_menu(self):
        menu = tk.Menu(self._root)

        menu_file = tk.Menu(menu, tearoff=0)
        menu_file.add_command(label="Import Data", command=self._handle_import)
        menu_file.add_command(label="Load Circuit", command=self._handle_load_circuit)
        menu_file.add_command(label="Save Circuit", command=self._handle_save_circuit)
        menu_file.add_separator()
        menu_file.add_command(label="Exit", command=None)
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

    def _read_cell_id(self):
        cell_id = self._cell_id_var.get()
        self._cell_id_var.set("")
        return cell_id

    def _handle_add_cell(self, event):
        if event.type == tk.EventType.ButtonPress:
            center = event.x, event.y
        else:
            center = (self._canvas.winfo_width() // 2, self._canvas.winfo_height() // 2)
        cell = Cell(self._read_cell_id())
        self._circuit.add_node(cell)
        Node(cell, self._canvas, center)

    def _handle_import(self):
        pass

    def _handle_map_estimation(self):
        pass

    def _handle_mcmc_sampling(self):
        pass

    def _handle_load_circuit(self):
        pass
        # filename = filedialog.askopenfilename(filetypes=[(f"{self.file_extension}", f"*.{self.file_extension}")])
        # if filename:
        #     self._graph.__del__()
        #     self._graph = pickle.load(open(filename, "rb"))
        #     self._graph.reload_circuit(self._canvas, self._connection_var)

    def _handle_save_circuit(self):
        pass
        # filename = filedialog.asksaveasfilename(filetypes=[(f"{self.file_extension}", f"*.{self.file_extension}")],
        #                                         defaultextension=f".{self.file_extension}")
        # if filename:
        #     pickle.dump(self._graph, open(filename, "wb"))