import tkinter as tk
from files.data_preprocessing.data_cleaning.data_service import DataService
from files.view.pages.start_frame_class import StartFrameClass
from files.view.pages.work_with_data_frame import Frame2
from tkinter import ttk

class App:
    def __init__(self, root, data_service: DataService):
        self.root = root
        self.root.title("Пример перехода между окнами")
        self.data_service = data_service
        self.frame1 = StartFrameClass(self, data_service)
        self.frame2 = Frame2(self)

        self.frame1.pack(fill='both', expand=True)

    def show_frame(self, frame):
        self.frame1.pack_forget()
        self.frame2.pack_forget()
        frame.pack(fill='both', expand=True)