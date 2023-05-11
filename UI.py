import sys
import os
import text_analyse as ta
import pandas as pd
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QComboBox, QPushButton, QLineEdit

class UIWindow(QWidget):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.selectedFile = None

        self.setGeometry(100, 100, 540, 480)
        self.setWindowTitle('HateSpeechDetectUI')

        self.file_label = QLabel(self)
        self.file_label.setText('Choose a file:')
        self.file_label.move(20, 20)

        self.file_label2 = QLabel(self)
        self.file_label2.setText('Or input text here:')
        self.file_label2.move(300, 20)

        self.file_dropdown = QComboBox(self)
        self.file_dropdown.move(20, 40)
        self.file_dropdown.resize(200, 25)
        self.file_dropdown.setPlaceholderText('No file selected')
        self.populateDropdown()
        self.file_dropdown.currentIndexChanged.connect(self.fileSelected)

        self.text_edit = QLineEdit(self)
        self.text_edit.move(300, 40)
        self.text_edit.resize(200, 25)

        self.button = QPushButton('Execute', self)
        self.button.move(20, 80)
        self.button.resize(100, 25)
        self.button.clicked.connect(self.dropDownExec)

        self.button2 = QPushButton('Execute', self)
        self.button2.move(300, 80)
        self.button2.resize(100, 25)
        self.button2.clicked.connect(self.textBoxExec)

        self.result_label = QLabel(self)
        self.result_label.move(20, 120)
        self.result_label.resize(620, 100)
        self.result_label.setText('Text:')

        self.result_label2 = QLabel(self)
        self.result_label2.move(20, 140)
        self.result_label2.resize(620, 100)

        self.result_label3 = QLabel(self)
        self.result_label3.move(20, 160)
        self.result_label3.resize(620, 100)
        self.result_label3.setText('Words identified:')

        self.result_label4 = QLabel(self)
        self.result_label4.move(20, 180)
        self.result_label4.resize(620, 100)

        self.result_percentage = QLabel(self)
        self.result_percentage.move(20, 200)
        self.result_percentage.resize(620, 100)
        self.result_percentage.setText('Hateful words percentage: ')

        self.show()

    def populateDropdown(self):
        folder_path = './data'
        files = os.listdir(folder_path)
        self.file_dropdown.addItems(files)

    def fileSelected(self, index):
        selected_file = self.file_dropdown.itemText(index)
        self.selectedFile = selected_file
        print(self.selectedFile)
        print(f'Selected file: {selected_file}')

    def textBoxExec(self):
        if self.text_edit.text() == '':
            return
        analyser = ta.analyseData('./model_old.pkl', self.text_edit.text(), False)
        df = analyser.sendResult()
        text = df[1]
        df = df[0]
        self.result_label4.setText(df.loc[0, 'text'])
        self.result_label2.setText(text)
        percentage = 'Hateful words percentage: ' + str(
            (len(df.loc[0, 'text'].split()) / len(text.split()) * 100)) + '%'
        self.result_percentage.setText(percentage)

    def dropDownExec(self):
        selected_file = self.file_dropdown.currentText()
        if self.selectedFile == None:
            return
        analyser = ta.analyseData('./models/gsdmm_models/model.pkl', './data/' + self.selectedFile, True)
        df = analyser.sendResult()
        text = df[1]
        df = df[0]
        self.result_label4.setText(df.loc[0, 'text'])
        self.result_label2.setText(text)
        percentage = 'Hateful words percentage: ' + str((len(df.loc[0, 'text'].split()) / len(text.split()) * 100)) + '%'
        self.result_percentage.setText(percentage)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = UIWindow()
    sys.exit(app.exec_())
