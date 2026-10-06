import os
import sys
import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QImage, QIcon
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QMessageBox,
    QVBoxLayout,
    QWidget,
    QFileDialog,
)

from pathlib import Path

root = Path(__file__).parent.absolute().__str__()
parent_root = Path(__file__).parent.parent.absolute().__str__()

sys.path.append(parent_root)

from inference_onnx import model_inference, post_process


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Bone Fracture Detection App")
        self.setGeometry(100, 100, 1200, 700)
        self.setWindowIcon(QIcon(os.path.join(root, "icons", "hand_radiography.png")))

        # ----------- STYLE SHEET -----------
        self.setStyleSheet("""
        QMainWindow {
            background-color: #f4f6f9;
        }

        QPushButton {
            background-color: #2d89ef;
            color: white;
            border-radius: 8px;
            padding: 6px;
            font-size: 14px;
            font-weight: bold;
        }

        QPushButton:hover {
            background-color: #1b5fbf;
        }

        QPushButton:disabled {
            background-color: #b0b0b0;
        }

        QLabel {
            font-size: 13px;
        }

        #imageLabel {
            border: 2px dashed #aaaaaa;
            background-color: white;
            border-radius: 10px;
        }
        """)

        # ----------- LAYOUTS -----------
        pagelayout = QHBoxLayout()
        button_layout = QVBoxLayout()
        right_layout = QVBoxLayout()
        images_layout = QHBoxLayout()
        labels_layout = QVBoxLayout()

        button_layout.setSpacing(15)
        button_layout.setContentsMargins(10, 20, 10, 20)

        # ----------- BUTTONS -----------

        btn_open_img = QPushButton("Open Image")
        btn_open_img.setIcon(QIcon(os.path.join(root, "icons", "folder.png")))
        btn_open_img.setFixedSize(162, 40)
        btn_open_img.pressed.connect(self.open_image)
        button_layout.addWidget(btn_open_img)

        self.btn_predict = QPushButton("Predict")
        self.btn_predict.setIcon(QIcon(os.path.join(root, "icons", "hand_radiography.png")))
        self.btn_predict.setFixedSize(162, 40)
        self.btn_predict.setEnabled(False)
        self.btn_predict.pressed.connect(self.predict)
        button_layout.addWidget(self.btn_predict)

        self.btn_save = QPushButton("Save")
        self.btn_save.setIcon(QIcon(os.path.join(root, "icons", "diskette.png")))
        self.btn_save.setFixedSize(162, 40)
        self.btn_save.setEnabled(False)
        self.btn_save.pressed.connect(self.save)
        button_layout.addWidget(self.btn_save)

        self.btn_clear = QPushButton("Clear")
        self.btn_clear.setIcon(QIcon(os.path.join(root, "icons", "restart.png")))
        self.btn_clear.setFixedSize(162, 40)
        self.btn_clear.setEnabled(False)
        self.btn_clear.pressed.connect(self.clear)
        button_layout.addWidget(self.btn_clear)

        btn_quit = QPushButton("Quit")
        btn_quit.setIcon(QIcon(os.path.join(root, "icons", "sign-out.png")))
        btn_quit.setFixedSize(162, 40)
        btn_quit.pressed.connect(self.quit)
        button_layout.addWidget(btn_quit)

        button_layout.addStretch()

        # ----------- IMAGE TITLES -----------

        title1 = QLabel("Original X-Ray")
        title1.setAlignment(Qt.AlignCenter)
        title1.setStyleSheet("font-size:16px; font-weight:bold;")

        title2 = QLabel("Fracture Detection")
        title2.setAlignment(Qt.AlignCenter)
        title2.setStyleSheet("font-size:16px; font-weight:bold;")

        # ----------- IMAGE PANELS -----------

        self.original_image = QLabel("Original Image")
        self.original_image.setAlignment(Qt.AlignCenter)
        self.original_image.setObjectName("imageLabel")
        self.original_image.setFixedSize(500, 500)

        self.predicted_image = QLabel("Prediction")
        self.predicted_image.setAlignment(Qt.AlignCenter)
        self.predicted_image.setObjectName("imageLabel")
        self.predicted_image.setFixedSize(500, 500)

        left_img_layout = QVBoxLayout()
        left_img_layout.addWidget(title1)
        left_img_layout.addWidget(self.original_image)

        right_img_layout = QVBoxLayout()
        right_img_layout.addWidget(title2)
        right_img_layout.addWidget(self.predicted_image)

        images_layout.addLayout(left_img_layout)
        images_layout.addLayout(right_img_layout)

        # ----------- LABELS -----------

        self.loaded_image_label = QLabel("No image loaded")
        self.loaded_image_label.setStyleSheet("font-size:14px; font-weight:bold;")

        self.save_log = QLabel("")
        self.save_log.setStyleSheet("color:green; font-size:13px;")

        labels_layout.addWidget(self.loaded_image_label)
        labels_layout.addWidget(self.save_log)

        # ----------- COMBINE LAYOUTS -----------

        right_layout.addLayout(images_layout)
        right_layout.addLayout(labels_layout)

        pagelayout.addLayout(button_layout)
        pagelayout.addLayout(right_layout)

        widget = QWidget()
        widget.setLayout(pagelayout)
        self.setCentralWidget(widget)

    # ----------- IMAGE CONVERTER -----------

    def _convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(500, 500, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    # ----------- OPEN IMAGE -----------

    def open_image(self):

        self.file_name = QFileDialog.getOpenFileName(
            self, "Open Image", parent_root, "Image Files (*.png *.jpg *.bmp)"
        )[0]

        if self.file_name == "":
            return

        self.img = cv2.imread(self.file_name)

        pixmap = self._convert_cv_qt(self.img)
        self.original_image.setPixmap(pixmap)

        self.btn_predict.setEnabled(True)
        self.btn_clear.setEnabled(True)
        self.btn_save.setEnabled(False)

        self.predicted_image.clear()
        self.save_log.clear()

        self.loaded_image_label.setText(
            f"Uploaded image: {os.path.basename(self.file_name)}"
        )

    # ----------- PREDICT -----------

    def predict(self):

        img = cv2.resize(self.img, (640, 640), interpolation=cv2.INTER_LINEAR) / 255
        img = np.asarray(img, dtype=np.float32)
        img = np.expand_dims(img, 0)
        img = img.transpose(0, 3, 1, 2)

        model_onnx_path = r'C:\Users\DELL\Desktop\bone_fracturte\yolov7-p6-bonefracture.onnx'

        output = model_inference(model_onnx_path, img)

        self.out_img, self.out_txt = post_process(self.file_name, output)

        pixmap = self._convert_cv_qt(self.out_img)
        self.predicted_image.setPixmap(pixmap)

        self.btn_save.setEnabled(True)

    # ----------- SAVE -----------

    def save(self):

        preds_dir = os.path.join(parent_root, "predictions")

        if os.path.exists(preds_dir) == False:
            os.makedirs(preds_dir, exist_ok=True)

        file_out_path = os.path.join(preds_dir, os.path.basename(self.file_name))

        save_file_name = QFileDialog.getSaveFileName(
            self, "Save File", file_out_path, "Images (*.png *.jpg)"
        )[0]

        if save_file_name == "":
            return

        cv2.imwrite(save_file_name, self.out_img)

        root = os.path.abspath(os.path.join(save_file_name, os.pardir))

        save_ann_file = os.path.join(
            root, os.path.basename(save_file_name)[:-4] + ".txt"
        )

        with open(save_ann_file, "w") as f:
            f.write(self.out_txt)

        self.save_log.setText(
            f"Predicted image saved in:\n{save_file_name}\nAnnotations saved in:\n{save_ann_file}"
        )

    # ----------- CLEAR -----------

    def clear(self):

        self.original_image.clear()
        self.predicted_image.clear()

        self.original_image.setText("Original Image")
        self.predicted_image.setText("Prediction")

        self.loaded_image_label.setText("No image loaded")
        self.save_log.clear()

        self.btn_predict.setEnabled(False)
        self.btn_save.setEnabled(False)
        self.btn_clear.setEnabled(False)

    # ----------- QUIT -----------

    def quit(self):

        msg = QMessageBox.question(
            self,
            "Close the app?",
            "Are you sure to close the app?",
            QMessageBox.Yes | QMessageBox.No,
        )

        if msg == QMessageBox.Yes:
            self.close()


app = QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec()