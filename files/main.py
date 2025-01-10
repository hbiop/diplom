from files.view.app import App
import tkinter as tk
from files.data_preprocessing.data_cleaning.data_service import DataService

if __name__ == "__main__":
    data_service = DataService()
    root = tk.Tk()
    app = App(root, data_service)
    root.mainloop()