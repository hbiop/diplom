import sys
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QPushButton,
    QFileDialog,
    QTableWidget,
    QTableWidgetItem,
    QLabel,
    QComboBox,
    QMessageBox,
    QTextEdit,
    QListView,
)

from sklearn.preprocessing import LabelEncoder
from files.data_preprocessing.data_cleaning.data_service import DataService



class MainWindow(QWidget):
    def __init__(self, ds: DataService):
        super().__init__()
        self.data_service : DataService = ds
        self.setWindowTitle("Data Preprocessing for Neural Network")
        self.setGeometry(100, 100, 800, 600)

        self.layout = QVBoxLayout()

        self.load_button = QPushButton("Load CSV Data")
        self.load_button.clicked.connect(self.load_data)
        self.layout.addWidget(self.load_button)
        self.delete_button = QPushButton("Удалить колонку")
        self.delete_button.clicked.connect(self.drop_columns)
        self.column_for_delete = QTextEdit()
        self.status_label = QLabel("")
        self.layout.addWidget(self.status_label)
        self.second_window_button = QPushButton("Подтвердить данные")
        self.second_window_button.clicked.connect(self.accept_data)
        self.table_widget = QTableWidget()
        self.layout.addWidget(self.table_widget)
        self.layout.addWidget(self.column_for_delete)
        self.layout.addWidget(self.delete_button)
        self.layout.addWidget(self.second_window_button)
        self.setLayout(self.layout)

    def load_data(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Open CSV File", "", "CSV Files (*.csv);;All Files (*)",
                                                   options=options)

        if file_name:
            try:
                self.data_service.read_data(file_name)
                self.status_label.setText(f"Loaded {file_name}")
                self.display_data()
            except Exception as e:
                self.status_label.setText(f"Error loading data: {e}")
    def accept_data(self):
        self.window = DataPreprocessingWindow(ds)
        self.window.show()
        self.close()


    def display_data(self):
        self.table_widget.clear()
        self.table_widget.setRowCount(len(self.data_service.data))
        self.table_widget.setColumnCount(len(self.data_service.data.columns))
        self.table_widget.setHorizontalHeaderLabels(self.data_service.data.columns)
        for i in range(len(self.data_service.data)):
            for j in range(len(self.data_service.data.columns)):
                self.table_widget.setItem(i, j, QTableWidgetItem(str(self.data_service.data.iat[i, j])))

    def drop_columns(self):
        self.data_service.drop_columns(self.column_for_delete.toPlainText())
        self.display_data()

class DataPreprocessingWindow(QWidget):
    def __init__(self, ds : DataService):
        super().__init__()
        self.data_service: DataService = ds
        self.setWindowTitle("Data Preprocessing for Neural Network")
        self.setGeometry(100, 100, 800, 600)
        self.layout = QVBoxLayout()
        self.table_widget = QTableWidget()
        self.button_next = QPushButton("Подтвердить")
        self.button_next.clicked.connect(self.accept_data)
        self.layout.addWidget(self.table_widget)
        self.layout.addWidget(self.button_next)
        self.setLayout(self.layout)
        self.data_service.prepare_data()
        self.display_data()

    def display_data(self):
        self.table_widget.clear()
        self.table_widget.setRowCount(len(self.data_service.data))
        self.table_widget.setColumnCount(len(self.data_service.data.columns))
        self.table_widget.setHorizontalHeaderLabels(self.data_service.data.columns)
        for i in range(len(self.data_service.data)):
            for j in range(len(self.data_service.data.columns)):
                self.table_widget.setItem(i, j, QTableWidgetItem(str(self.data_service.data.iat[i, j])))

    def accept_data(self):
        self.window = NeuralNetworkScreen()
        self.window.show()
        self.close()

class NeuralNetworkScreen(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Neural network settings")
        self.setGeometry(100, 100, 800, 600)
        self.label_count_layers = QPushButton("Добавить слой")
        self.combobox_count_layers = QListView()
        self.label_count_neurons = QLabel("Введите количество нейронов")
        self.combobox_count_neurons = QComboBox()
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.label_count_layers)
        self.layout.addWidget(self.combobox_count_layers)
        self.layout.addWidget(self.label_count_neurons)
        self.layout.addWidget(self.combobox_count_neurons)
        self.setLayout(self.layout)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ds = DataService()
    window = MainWindow(ds)
    window.show()
    sys.exit(app.exec_())