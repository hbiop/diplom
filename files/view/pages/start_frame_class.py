from cgitb import enable

import pandas as pd
import numpy as np
import tkinter as tk
from files.data_preprocessing.data_cleaning.data_service import DataService
from tkinter import filedialog, messagebox
from tkinter import ttk

class StartFrameClass(tk.Frame):
    def __init__(self, controller, data_sevice:DataService):
        super().__init__(controller.root)

        self.load_button = tk.Button(self, text="Загрузить CSV файл", command=self.load_file)
        self.load_button.pack(pady=10)

        self.data_service = data_sevice

        self.label = tk.Label(self, text="Ваши данные:")
        self.label.pack(pady=10)

        self.tree = ttk.Treeview(self)
        self.tree.pack(expand=True, fill='both')

        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview)
        self.scrollbar.pack(side='right', fill='y')
        self.tree.configure(yscrollcommand=self.scrollbar.set)

        self.button_continue = tk.Button(self,state=["disabled"],text="Перейти к обработке данных", command=lambda: controller.show_frame(controller.frame2))
        self.button_continue.pack(pady=10)

    def load_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            try:
                self.data_service.read_data(file_path)
                self.data_service.print_data()
                self.display_data()
            except Exception as e:
                messagebox.showerror("Ошибка", str(e))

    def display_data(self):
        for i in self.tree.get_children():
            self.tree.delete(i)
        self.tree["columns"] = list(self.data_service.data)
        self.tree["show"] = "headings"

        for column in self.data_service.data.columns:
            self.tree.heading(column, text=column)  # Название столбца
            self.tree.column(column, anchor="center")  # Выравнивание столбца

        for index, row in self.data_service.data.iterrows():
            self.tree.insert("", "end", values=list(row))
        self.button_continue.configure(state="normal")

