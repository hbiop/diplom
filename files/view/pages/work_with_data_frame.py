import tkinter as tk



class Frame2(tk.Frame):
    def __init__(self, controller):
        super().__init__(controller.root)
        label = tk.Label(self, text="Это второй экран")
        label.pack(pady=20)

        button = tk.Button(self, text="Вернуться на первый экран",
                           command=lambda: controller.show_frame(controller.frame1))
        button.pack(pady=10)