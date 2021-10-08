import os
import sys
import logging
import shutil

from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QLineEdit, QGridLayout, QFileDialog, QCheckBox,\
                            QMessageBox, QGroupBox
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

        self.flats_darks_group = QGroupBox("Use Common Set of Flats and Darks")
        self.flats_darks_group.clicked.connect(self.set_flats_darks_group)

        self.flats_button = QPushButton("Select Flats Path")
        self.flats_button.clicked.connect(self.flats_button_pressed)
        self.flats_entry = QLineEdit()
        self.flats_entry.textChanged.connect(self.set_flats_entry)

        self.darks_button = QPushButton("Select Darks Path")
        self.darks_button.clicked.connect(self.darks_button_pressed)
        self.darks_entry = QLineEdit()
        self.darks_entry.textChanged.connect(self.set_darks_entry)

        self.overlap_region_label = QLabel("Overlap Region Size")
        self.overlap_region_entry = QLineEdit()
        self.overlap_region_entry.textChanged.connect(self.set_overlap_region_entry)

        self.left_hand_checkbox = QCheckBox("Is the rotation axis on the left-hand side of the image?")
        self.left_hand_checkbox.stateChanged.connect(self.set_left_hand_checkbox)

        self.help_button = QPushButton("Help")
        self.help_button.clicked.connect(self.help_button_pressed)

        self.delete_temp_button = QPushButton("Delete Output Directory")
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

        self.flats_darks_group.setCheckable(True)
        self.flats_darks_group.setChecked(False)
        flats_darks_layout = QGridLayout()
        flats_darks_layout.addWidget(self.flats_button, 0, 0, 1, 2)
        flats_darks_layout.addWidget(self.flats_entry, 0, 2, 1, 2)
        flats_darks_layout.addWidget(self.darks_button, 1, 0, 1, 2)
        flats_darks_layout.addWidget(self.darks_entry, 1, 2, 1, 2)
        self.flats_darks_group.setLayout(flats_darks_layout)
        layout.addWidget(self.flats_darks_group, 2, 0, 1, 4)

        layout.addWidget(self.overlap_region_label, 3, 2)
        layout.addWidget(self.overlap_region_entry, 3, 3)
        layout.addWidget(self.left_hand_checkbox, 3, 0, 1, 2)
        layout.addWidget(self.stitch_button, 4, 0, 1, 2)
        layout.addWidget(self.help_button, 4, 3, 1, 3)
        layout.addWidget(self.delete_temp_button, 4, 2, 1, 1)
        self.setLayout(layout)

    def init_values(self):
        self.input_entry.setText("...enter input directory")
        self.output_entry.setText("...enter output directory")
        self.flats_entry.setText("...enter flats directory")
        self.parameters['flats_dir'] = ""
        self.darks_entry.setText("...enter darks directory")
        self.parameters['darks_dir'] = ""
        self.overlap_region_entry.setText("770")
        self.parameters['overlap_region'] = "770"
        self.left_hand_checkbox.setChecked(False)
        self.parameters['axis_on_left'] = str(False)

    def input_button_pressed(self):
        logging.debug("Input Button Pressed")
        dir_explore = QFileDialog(self)
        input_dir = dir_explore.getExistingDirectory()
        self.input_entry.setText(input_dir)
        self.parameters['input_dir'] = input_dir

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

    def set_flats_darks_group(self):
        logging.debug("Use Common Flats/Darks: " + str(self.flats_darks_group.isChecked()))

    def flats_button_pressed(self):
        logging.debug("Flats Button Pressed")
        dir_explore = QFileDialog(self)
        flats_dir = dir_explore.getExistingDirectory()
        self.flats_entry.setText(flats_dir)
        self.parameters['flats_dir'] = flats_dir

    def set_flats_entry(self):
        logging.debug("Flats Entry: " + str(self.flats_entry.text()))
        self.parameters['flats_dir'] = str(self.flats_entry.text())

    def darks_button_pressed(self):
        logging.debug("Darks Button Pressed")
        dir_explore = QFileDialog(self)
        darks_dir = dir_explore.getExistingDirectory()
        self.darks_entry.setText(darks_dir)
        self.parameters['darks_dir'] = darks_dir

    def set_darks_entry(self):
        logging.debug("Darks Entry: " + str(self.darks_entry.text()))
        self.parameters['darks_dir'] = str(self.darks_entry.text())

    def set_overlap_region_entry(self):
        logging.debug("Overlap Region: " + str(self.overlap_region_entry.text()))
        self.parameters['overlap_region'] = str(self.overlap_region_entry.text())

    def set_left_hand_checkbox(self):
        logging.debug("Rotation axis on left-hand-side checkbox: " + str(self.left_hand_checkbox.isChecked()))
        self.parameters['axis_on_left'] = str(self.left_hand_checkbox.isChecked())

    def help_button_pressed(self):
        logging.debug("Help Button Pressed")
        h = "Lorem Ipsum\n"
        QMessageBox.information(self, "Help", h)

    def delete_button_pressed(self):
        logging.debug("Delete Output Directory Button Pressed")
        delete_dialog = QMessageBox.question(self, 'Quit', 'Are you sure you want to delete the temporary directory?',
                                             QMessageBox.Yes | QMessageBox.No)
        if delete_dialog == QMessageBox.Yes:
            try:
                print("Deleting: " + self.parameters['input_dir'] + " ...")
                shutil.rmtree(self.parameters['input_dir'])
                print("Deleted directory: " + self.parameters['input_dir'])
            except FileNotFoundError:
                print("Directory does not exist: " + self.parameters['input_dir'])

    def stitch_button_pressed(self):
        logging.debug("Stitch Button Pressed")
        self.auto_stitch_funcs = AutoStitchFunctions(self.parameters)
        if not os.path.isdir(self.parameters['output_dir']):
            self.auto_stitch_funcs.run_auto_stitch()
        else:
            print("--> Temp Directory Exists - Delete Before Proceeding")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = AutoStitchGUI()
    sys.exit(app.exec_())

