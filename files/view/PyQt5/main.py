from PyQt5.QtGui import QStandardItem, QStandardItemModel
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
    QTextEdit,
    QListView, QMainWindow,
)
from forms.neural_network_settings.neural_network_settings import Ui_MainWindow
from files.neural_network.layers.layer import Layer
from PyQt5 import uic
import sys
from files.data_preprocessing.data_cleaning.data_service import DataService
#qt5-tools designer

class MainWindow(QWidget):
    def __init__(self, ds: DataService):
        super().__init__()
        self.data_service : DataService = ds
        self.setWindowTitle("Обработка данных для нейронной сети")
        self.setGeometry(100, 100, 800, 600)

        self.layout = QVBoxLayout()

        self.load_button = QPushButton("Загрузить данные")
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
        self.setWindowTitle("Предпросмотр данных")
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
        self.window = SecondWindow(self.data_service)
        self.window.show()
        self.close()


class SecondWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, ds: DataService):
        super(SecondWindow,self).__init__()
        self.setupUi(self)
        self.data_service = ds
        self.ButtonAdd.clicked.connect(self.add_layer)
        self.SpinBoxBatchSize.setValue(1)
        self.SpinBoxBatchSize.valueChanged.connect(self.spin_batch_size_changed)
        self.layers : list[Layer] = []
        self.model = QStandardItemModel(self)
        self.ListViewNeuralNetworkLayers.setModel(self.model)

        for i in range(100):
            item = QStandardItem(f'Элемент {i}')
            self.model.appendRow(item)
    def add_layer(self):
        print("connect")

    def spin_batch_size_changed(self):
        if self.SpinBoxBatchSize.value() > 10:
            self.SpinBoxBatchSize.setValue(10)
        if self.SpinBoxBatchSize.value() < 1:
            self.SpinBoxBatchSize.setValue(1)


class MyApp(QWidget):
    def __init__(self):
        super().__init__()
        uic.loadUi("forms/neural_network_settings.ui", self)

    def add_layer(self):
        layer_name = f'Слой {len(self.layers) + 1}'
        params = f'Параметры: вес = {len(self.layers) * 10}, смещение = {len(self.layers) * 5}'
        new_layer = Layer(layer_name, params)
        self.layers.append(new_layer)
        item = QStandardItem(new_layer.name)
        self.model.appendRow(item)

    def display_layer_info(self, index):
        if index.isValid():
            selected_layer = self.layers[index.row()]
            self.info_text.setText(f'Имя: {selected_layer.name}n{selected_layer.params}')
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