import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton

class Example(QMainWindow):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout()
        central_widget.setLayout(layout)

        self.labels = []

        # 添加标签
        for i in range(5):
            label = QLabel(f"Label {i+1}")
            layout.addWidget(label)
            self.labels.append(label)

        # 添加清除按钮
        clear_button = QPushButton("Clear Labels")
        clear_button.clicked.connect(self.clearLabels)
        layout.addWidget(clear_button)

        self.setGeometry(300, 300, 300, 200)
        self.setWindowTitle('Clear Labels Example')
        self.show()

    def clearLabels(self):
        # 清除所有标签的文本
        for label in self.labels:
            label.setText("")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())
