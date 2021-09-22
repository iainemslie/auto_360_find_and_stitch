import os
import sys
import logging
import shutil

from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QLineEdit, QGridLayout, QFileDialog, QCheckBox,\
                            QMessageBox
from auto_stitch_funcs import AutoStitchFunctions


class AutoStitchGUI(QWidget):

    def __init__(self, *args, **kwargs):
        super(AutoStitchGUI, self).__init__(*args, **kwargs)
        self.setWindowTitle('Auto Stitch')

        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)

        self.parameters = {}
        self.auto_stitch_funcs = None

        self.input_button = QPushButton("Select Input Path")
        self.input_button.clicked.connect(self.input_button_pressed)
        self.input_entry = QLineEdit()
        self.input_entry.textChanged.connect(self.set_input_entry)

        self.output_button = QPushButton("Select Output Path")
        self.output_button.clicked.connect(self.output_button_pressed)
        self.output_entry = QLineEdit()
        self.output_entry.textChanged.connect(self.set_output_entry)

        self.temp_button = QPushButton("Select Temp Path")
        self.temp_button.clicked.connect(self.temp_button_pressed)
        self.temp_entry = QLineEdit()
        self.temp_entry.textChanged.connect(self.set_temp_entry)

        self.overlap_start_label = QLabel("Starting horizontal overlap")
        self.overlap_start_entry = QLineEdit()
        self.overlap_start_entry.textChanged.connect(self.set_overlap_start_entry)

        self.overlap_end_label = QLabel("Ending horizontal overlap")
        self.overlap_end_entry = QLineEdit()
        self.overlap_end_entry.textChanged.connect(self.set_overlap_end_entry)

        self.steps_label = QLabel("Number of steps")
        self.steps_entry = QLineEdit()
        self.steps_entry.textChanged.connect(self.set_steps_entry)

        self.left_hand_checkbox = QCheckBox("Is the rotation axis on the left-hand side of the image?")
        self.left_hand_checkbox.stateChanged.connect(self.set_left_hand_checkbox)

        self.help_button = QPushButton("Help")
        self.help_button.clicked.connect(self.help_button_pressed)

        self.delete_temp_button = QPushButton("Delete Temp Directory")
        self.delete_temp_button.clicked.connect(self.delete_button_pressed)

        self.stitch_button = QPushButton("Stitch")
        self.stitch_button.clicked.connect(self.stitch_button_pressed)

        self.set_layout()
        self.resize(800, 0)

        self.init_values()
        self.show()

    def set_layout(self):
        layout = QGridLayout()
        layout.addWidget(self.input_button, 0, 0, 1, 2)
        layout.addWidget(self.input_entry, 0, 2, 1, 4)
        layout.addWidget(self.output_button, 1, 0, 1, 2)
        layout.addWidget(self.output_entry, 1, 2, 1, 4)
        layout.addWidget(self.temp_button, 2, 0, 1, 2)
        layout.addWidget(self.temp_entry, 2, 2, 1, 4)
        layout.addWidget(self.overlap_start_label, 3, 0)
        layout.addWidget(self.overlap_start_entry, 3, 1)
        layout.addWidget(self.overlap_end_label, 3, 2)
        layout.addWidget(self.overlap_end_entry, 3, 3)
        layout.addWidget(self.steps_label, 3, 4)
        layout.addWidget(self.steps_entry, 3, 5)
        layout.addWidget(self.left_hand_checkbox, 4, 0, 1, 4)
        layout.addWidget(self.help_button, 5, 0, 1, 2)
        layout.addWidget(self.delete_temp_button, 5, 2, 1, 1)
        layout.addWidget(self.stitch_button, 5, 3, 1, 3)
        self.setLayout(layout)

    def init_values(self):
        working_dir = os.getcwd()
        self.input_entry.setText(working_dir)
        self.parameters['input_dir'] = working_dir
        output_dir = working_dir + "/rec/auto_horstitch"
        self.output_entry.setText(output_dir)
        self.parameters['output_dir'] = output_dir
        temp_dir = "/data/tmp-auto-stitch"
        self.temp_entry.setText(temp_dir)
        self.parameters['temp_dir'] = temp_dir
        self.overlap_start_entry.setText("720")
        self.parameters['overlap_start'] = "720"
        self.overlap_end_entry.setText("735")
        self.parameters['overlap_end'] = "735"
        self.steps_entry.setText("1")
        self.parameters['steps'] = "1"
        self.left_hand_checkbox.setChecked(False)
        self.parameters['axis_on_left'] = str(False)

    def input_button_pressed(self):
        logging.debug("Input Button Pressed")
        dir_explore = QFileDialog(self)
        input_dir = dir_explore.getExistingDirectory()
        self.input_entry.setText(input_dir)
        self.parameters['input_dir'] = input_dir
        self.output_entry.setText(input_dir + "/rec/horstitch")
        self.parameters['output_dir'] = input_dir + "/rec/horstitch"

    def set_input_entry(self):
        logging.debug("Input Entry: " + str(self.input_entry.text()))
        self.parameters['input_dir'] = str(self.input_entry.text())

    def output_button_pressed(self):
        logging.debug("Output Button Pressed")
        dir_explore = QFileDialog(self)
        output_dir = dir_explore.getExistingDirectory()
        self.output_entry.setText(output_dir)
        self.parameters['output_dir'] = output_dir

    def set_output_entry(self):
        logging.debug("Output Entry: " + str(self.output_entry.text()))
        self.parameters['output_dir'] = str(self.output_entry.text())

    def temp_button_pressed(self):
        logging.debug("Temp Button Pressed")
        dir_explore = QFileDialog(self)
        temp_dir = dir_explore.getExistingDirectory()
        self.temp_entry.setText(temp_dir)
        self.parameters['temp_dir'] = temp_dir

    def set_temp_entry(self):
        logging.debug("Temp Entry: " + str(self.temp_entry.text()))
        self.parameters['temp_dir'] = str(self.temp_entry.text())

    def set_overlap_start_entry(self):
        logging.debug("Overlap Start: " + str(self.overlap_start_entry.text()))
        self.parameters['overlap_start'] = str(self.overlap_start_entry.text())

    def set_overlap_end_entry(self):
        logging.debug("Overlap End: " + str(self.overlap_end_entry.text()))
        self.parameters['overlap_end'] = str(self.overlap_end_entry.text())

    def set_steps_entry(self):
        logging.debug("Steps: " + str(self.steps_entry.text()))
        self.parameters['steps'] = str(self.steps_entry.text())

    def set_left_hand_checkbox(self):
        logging.debug("Rotation axis on left-hand-side checkbox: " + str(self.left_hand_checkbox.isChecked()))
        self.parameters['axis_on_left'] = str(self.left_hand_checkbox.isChecked())

    def help_button_pressed(self):
        logging.debug("Help Button Pressed")
        h = "Lorem Ipsum\n"
        QMessageBox.information(self, "Help", h)

    def delete_button_pressed(self):
        logging.debug("Delete Temp Directory Button Pressed")
        delete_dialog = QMessageBox.question(self, 'Quit', 'Are you sure you want to delete the temporary directory?',
                                             QMessageBox.Yes | QMessageBox.No)
        if delete_dialog == QMessageBox.Yes:
            try:
                print("Deleting: " + self.parameters['temp_dir'] + " ...")
                shutil.rmtree(self.parameters['temp_dir'])
                print("Deleted directory: " + self.parameters['temp_dir'])
            except FileNotFoundError:
                print("Directory does not exist: " + self.parameters['temp_dir'])

    def stitch_button_pressed(self):
        logging.debug("Stitch Button Pressed")
        self.auto_stitch_funcs = AutoStitchFunctions(self.parameters)
        self.auto_stitch_funcs.run_auto_stitch()
        if not os.path.isdir(self.parameters['temp_dir']):
            self.auto_stitch_funcs.run_auto_stitch()
        else:
            print("--> Temp Directory Exists - Delete Before Proceeding")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = AutoStitchGUI()
    sys.exit(app.exec_())

