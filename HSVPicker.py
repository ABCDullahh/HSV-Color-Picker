import sys
import os
import cv2
import numpy as np
import pyautogui
import win32gui

from sklearn.cluster import KMeans
from collections import defaultdict

from PyQt5 import QtGui
from PyQt5.QtCore import Qt, QSize, QTimer
from PyQt5.QtGui import (
    QColor, QImage, QPixmap,
    QPainter, QPen, QGuiApplication
)
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QPushButton, QSlider, QCheckBox,
    QComboBox, QFileDialog, QSplitter
)


def generateSolidColorPixmap(w, h, color):
    """Membuat QImage berukuran (w,h) dengan warna solid."""
    canvas = QImage(QSize(w, h), QImage.Format_RGB30)
    for baris in range(h):
        for kolom in range(w):
            canvas.setPixel(kolom, baris, color.rgb())
    return canvas


class ImprovedHSVPicker(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Improved HSV Color Picker")

        # Variabel internal
        self.selectedHue = 0
        self.selectedSaturation = 255
        self.selectedValue = 255
        self.lowerHSV = (0, 0, 0)
        self.upperHSV = (179, 255, 255)

        self.imgRaw = None
        self.imgHsvSpace = None

        self.is_picking_color = False
        self.is_sampling = False
        self.selected_points = []

        # Zoom preview
        self.zoom_factor = 8
        self.zoom_window_size = 10

        # Bangun UI
        self.setup_ui()
        self.init_handler()
        self.loadHsvSpace()
        self.updateHSVPreview()

        # Timer pick color
        self.color_picker_timer = QTimer()
        self.color_picker_timer.timeout.connect(self.update_color_from_screen)

    def setup_ui(self):
        """
        Mengatur layout utama aplikasi dengan QSplitter
        agar panel kiri & kanan dapat di-resize secara fleksibel.
        """
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # QSplitter sebagai pemisah area kiri dan kanan
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # Buat container untuk sisi kiri dan kanan
        left_container = QWidget()
        right_container = QWidget()
        splitter.addWidget(left_container)
        splitter.addWidget(right_container)

        # Layout vertikal untuk sisi kiri
        left_column = QVBoxLayout(left_container)
        left_column.setContentsMargins(10, 10, 10, 10)
        left_column.setSpacing(10)

        # Layout vertikal untuk sisi kanan
        right_column = QVBoxLayout(right_container)
        right_column.setContentsMargins(10, 10, 10, 10)
        right_column.setSpacing(10)

        # ================== PANEL KIRI ==================
        # 1) HSV Controls
        hsv_group = QGroupBox("HSV Controls")
        hsv_layout = QVBoxLayout()
        hsv_layout.setContentsMargins(5, 5, 5, 5)

        self.sliderH = QSlider(Qt.Horizontal)
        self.sliderS = QSlider(Qt.Horizontal)
        self.sliderV = QSlider(Qt.Horizontal)

        self.sliderH.setRange(0, 359)
        self.sliderS.setRange(0, 255)
        self.sliderV.setRange(0, 255)

        self.lblH = QLabel("H: 0")
        self.lblS = QLabel("S: 255")
        self.lblV = QLabel("V: 255")

        for slider, label in [
            (self.sliderH, self.lblH),
            (self.sliderS, self.lblS),
            (self.sliderV, self.lblV)
        ]:
            row_layout = QHBoxLayout()
            row_layout.addWidget(label, 1)
            row_layout.addWidget(slider, 4)
            hsv_layout.addLayout(row_layout)

        hsv_group.setLayout(hsv_layout)
        left_column.addWidget(hsv_group)

        # 2) Preview Warna (H, S, V)
        preview_group = QGroupBox("Color Preview")
        preview_layout = QHBoxLayout()
        preview_layout.setContentsMargins(5, 5, 5, 5)

        self.previewH = QLabel()
        self.previewS = QLabel()
        self.previewV = QLabel()

        for preview_label in [self.previewH, self.previewS, self.previewV]:
            preview_label.setFixedSize(80, 80)
            preview_layout.addWidget(preview_label)

        preview_group.setLayout(preview_layout)
        left_column.addWidget(preview_group)

        # 3) Range Control (LOWER/UPPER)
        range_group = QGroupBox("HSV Range")
        range_layout = QVBoxLayout()
        range_layout.setContentsMargins(5, 5, 5, 5)

        self.cboxSetMode = QComboBox()
        self.cboxSetMode.addItems(["LOWER", "UPPER"])

        self.lblLower = QLabel("Lower: H 0; S 0; V 0")
        self.lblUpper = QLabel("Upper: H 179; S 255; V 255")

        range_layout.addWidget(self.cboxSetMode)
        range_layout.addWidget(self.lblLower)
        range_layout.addWidget(self.lblUpper)
        range_group.setLayout(range_layout)
        left_column.addWidget(range_group)

        # 4) HSV Space Preview
        hsv_space_group = QGroupBox("HSV Color Space")
        hsv_space_layout = QVBoxLayout()
        hsv_space_layout.setContentsMargins(5, 5, 5, 5)

        self.previewHsvSpace = QLabel()
        self.previewHsvSpace.setFixedSize(300, 120)
        hsv_space_layout.addWidget(self.previewHsvSpace)
        hsv_space_group.setLayout(hsv_space_layout)
        left_column.addWidget(hsv_space_group)

        # 5) Tombol Fungsi
        button_group = QGroupBox("Main Actions")
        button_layout = QHBoxLayout()
        button_layout.setSpacing(5)

        self.btnOpen = QPushButton("Open Image")
        self.btnPickColor = QPushButton("Pick Color")
        self.btnCopy = QPushButton("Copy Range")
        self.btnSample = QPushButton("Sample Points")
        self.btnAutoTune = QPushButton("Auto Tune HSV")

        for btn in [self.btnOpen, self.btnPickColor, self.btnCopy, self.btnSample, self.btnAutoTune]:
            button_layout.addWidget(btn)

        button_group.setLayout(button_layout)
        left_column.addWidget(button_group)

        # 6) Zoom Preview
        zoom_group = QGroupBox("Zoom Preview")
        zoom_layout = QVBoxLayout()
        zoom_layout.setContentsMargins(5, 5, 5, 5)

        self.zoomLabel = QLabel()
        self.zoomLabel.setFixedSize(220, 160)
        self.zoomLabel.setStyleSheet("border: 1px solid gray;")
        zoom_layout.addWidget(self.zoomLabel)
        zoom_group.setLayout(zoom_layout)
        left_column.addWidget(zoom_group)

        # ================== PANEL KANAN ==================
        # Image Preview
        img_preview_group = QGroupBox("Image Preview")
        img_preview_layout = QVBoxLayout()
        img_preview_layout.setContentsMargins(5, 5, 5, 5)

        # Perbesar preview menjadi 640×360
        preview_size = QSize(640, 360)
        self.previewRaw = QLabel()
        self.previewMask = QLabel()
        self.previewMaskedRaw = QLabel()

        for preview_label in [self.previewRaw, self.previewMask, self.previewMaskedRaw]:
            preview_label.setFixedSize(preview_size)
            preview_label.setAlignment(Qt.AlignCenter)
            preview_label.setStyleSheet("border: 1px solid #ccc;")
            img_preview_layout.addWidget(preview_label)

        # Morphology Controls
        morph_group = QGroupBox("Morphology")
        morph_layout = QHBoxLayout()

        self.cboxErode = QCheckBox("Erode")
        self.sliderErotion = QSlider(Qt.Horizontal)
        self.sliderErotion.setRange(1, 20)

        self.cboxDilate = QCheckBox("Dilate")
        self.sliderDilation = QSlider(Qt.Horizontal)
        self.sliderDilation.setRange(1, 20)

        morph_layout.addWidget(self.cboxErode)
        morph_layout.addWidget(self.sliderErotion)
        morph_layout.addWidget(self.cboxDilate)
        morph_layout.addWidget(self.sliderDilation)
        morph_group.setLayout(morph_layout)

        img_preview_layout.addWidget(morph_group)
        img_preview_group.setLayout(img_preview_layout)
        right_column.addWidget(img_preview_group)

        # -- Aktifkan Mouse Tracking di previewRaw --
        self.previewRaw.setMouseTracking(True)
        # Dengan ini, setiap gerakan mouse di area previewRaw akan memicu event mouseMoveEvent

    def init_handler(self):
        # Handler slider
        self.sliderH.valueChanged.connect(self.onHChanged)
        self.sliderS.valueChanged.connect(self.onSChanged)
        self.sliderV.valueChanged.connect(self.onVChanged)

        # Combobox
        self.cboxSetMode.currentTextChanged.connect(self.onCBoxModeChanged)

        # Tombol
        self.btnOpen.clicked.connect(self.onBtnOpenClicked)
        self.btnPickColor.clicked.connect(self.toggle_color_picker)
        self.btnCopy.clicked.connect(self.onBtnCopyClicked)
        self.btnSample.clicked.connect(self.toggle_sampling)
        self.btnAutoTune.clicked.connect(self.auto_tune_hsv)

        # Mouse click pada previewRaw (untuk sampling)
        self.previewRaw.mousePressEvent = self.on_image_click
        # Mouse move untuk realtime zoom
        self.previewRaw.mouseMoveEvent = self.on_image_mouse_move

        # Morphology
        self.cboxErode.stateChanged.connect(self.updateMask)
        self.cboxDilate.stateChanged.connect(self.updateMask)
        self.sliderErotion.valueChanged.connect(self.onSliderErodeChanged)
        self.sliderDilation.valueChanged.connect(self.onSliderDilateChanged)

    # ============== ZOOM PREVIEW ==============
    def showZoomPreview(self, x, y):
        if self.imgRaw is None:
            return
        h, w, _ = self.imgRaw.shape
        half = self.zoom_window_size

        x1 = max(0, x - half)
        x2 = min(w, x + half)
        y1 = max(0, y - half)
        y2 = min(h, y + half)

        roi = self.imgRaw[y1:y2, x1:x2].copy()
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

        zoomed = cv2.resize(
            roi_rgb, None,
            fx=self.zoom_factor,
            fy=self.zoom_factor,
            interpolation=cv2.INTER_NEAREST
        )

        zh, zw, _ = zoomed.shape
        cx = zw // 2
        cy = zh // 2
        cross_color = (0, 255, 0)

        cv2.line(zoomed, (0, cy), (zw, cy), cross_color, 1)
        cv2.line(zoomed, (cx, 0), (cx, zh), cross_color, 1)
        cv2.circle(zoomed, (cx, cy), 5, cross_color, 1)

        qimg = QImage(
            zoomed.data,
            zoomed.shape[1],
            zoomed.shape[0],
            zoomed.shape[1]*3,
            QImage.Format_RGB888
        )
        pixmap = QPixmap.fromImage(qimg)
        self.zoomLabel.setPixmap(
            pixmap.scaled(
                self.zoomLabel.width(),
                self.zoomLabel.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
        )

    def on_image_mouse_move(self, event):
        """
        Event handler saat mouse bergerak di atas previewRaw.
        Kita gunakan untuk menampilkan zoom preview secara real-time.
        """
        if self.imgRaw is None:
            return

        # Koordinat mouse di label previewRaw
        mouse_x = event.pos().x()
        mouse_y = event.pos().y()

        # Pastikan skala sudah tersimpan (setelah open image)
        if not hasattr(self, 'preview_scale'):
            return

        relative_x = mouse_x - self.preview_scale['offset_x']
        relative_y = mouse_y - self.preview_scale['offset_y']

        scale_x = (
            self.preview_scale['original_size'][0]
            / self.preview_scale['preview_size'][0]
        )
        scale_y = (
            self.preview_scale['original_size'][1]
            / self.preview_scale['preview_size'][1]
        )

        img_x = int(relative_x * scale_x)
        img_y = int(relative_y * scale_y)

        # Pastikan masih di dalam range gambar
        if (0 <= img_x < self.imgRaw.shape[1]) and (0 <= img_y < self.imgRaw.shape[0]):
            self.showZoomPreview(img_x, img_y)

    # ============== SAMPLING ==============
    def toggle_sampling(self):
        self.is_sampling = not self.is_sampling
        if self.is_sampling:
            self.btnSample.setText("Stop Sampling")
            self.selected_points.clear()
        else:
            self.btnSample.setText("Sample Points")
            print(f"Collected {len(self.selected_points)} sample points")

    def on_image_click(self, event):
        if not self.is_sampling or self.imgRaw is None:
            return

        mouse_x = event.pos().x()
        mouse_y = event.pos().y()

        relative_x = mouse_x - self.preview_scale['offset_x']
        relative_y = mouse_y - self.preview_scale['offset_y']

        scale_x = self.preview_scale['original_size'][0] / self.preview_scale['preview_size'][0]
        scale_y = self.preview_scale['original_size'][1] / self.preview_scale['preview_size'][1]

        img_x = int(relative_x * scale_x)
        img_y = int(relative_y * scale_y)

        if (
            0 <= img_x < self.imgRaw.shape[1]
            and 0 <= img_y < self.imgRaw.shape[0]
            and 0 <= relative_x < self.preview_scale['preview_size'][0]
            and 0 <= relative_y < self.preview_scale['preview_size'][1]
        ):
            hsv = cv2.cvtColor(self.imgRaw, cv2.COLOR_BGR2HSV)
            point_hsv = hsv[img_y, img_x]
            self.selected_points.append((img_x, img_y))
            print(f"Added sample point at ({img_x}, {img_y}): HSV{tuple(point_hsv)}")

            self.draw_sample_points()
            # Tampilkan zoom preview juga saat klik
            self.showZoomPreview(img_x, img_y)

    def draw_sample_points(self):
        if self.imgRaw is None or not self.selected_points:
            return

        current_pixmap = self.previewRaw.pixmap()
        if current_pixmap is None:
            return

        temp_pixmap = QPixmap(current_pixmap)
        painter = QPainter(temp_pixmap)
        painter.setRenderHint(QPainter.Antialiasing)

        pen = QPen(QColor(0, 255, 0))
        pen.setWidth(2)
        painter.setPen(pen)

        scale_x = self.preview_scale['preview_size'][0] / self.preview_scale['original_size'][0]
        scale_y = self.preview_scale['preview_size'][1] / self.preview_scale['original_size'][1]

        for x, y in self.selected_points:
            px = int(x * scale_x) + self.preview_scale['offset_x']
            py = int(y * scale_y) + self.preview_scale['offset_y']

            size = 10
            painter.drawLine(px - size, py, px + size, py)
            painter.drawLine(px, py - size, px, py + size)
            painter.drawEllipse(px - 5, py - 5, 10, 10)

        painter.end()
        self.previewRaw.setPixmap(temp_pixmap)

    # ============== AUTO TUNE ==============
    def auto_tune_hsv(self):
        if not self.selected_points:
            print("No sample points collected!")
            return

        hsv_img = cv2.cvtColor(self.imgRaw, cv2.COLOR_BGR2HSV)
        samples = []
        for x, y in self.selected_points:
            samples.append(hsv_img[y, x])
        samples = np.array(samples)

        if len(samples) == 0:
            print("No valid samples!")
            return

        n_clusters = min(3, len(samples))
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(samples)

        ranges = defaultdict(lambda: {'min': float('inf'), 'max': float('-inf')})
        for cluster_idx in range(n_clusters):
            cluster_points = samples[kmeans.labels_ == cluster_idx]
            if len(cluster_points) == 0:
                continue

            for i, comp in enumerate(['h', 's', 'v']):
                comp_values = cluster_points[:, i]
                # Gunakan percentile untuk memfilter outlier
                ranges[comp]['min'] = min(ranges[comp]['min'], np.percentile(comp_values, 5))
                ranges[comp]['max'] = max(ranges[comp]['max'], np.percentile(comp_values, 95))

        h_margin = 10
        s_margin = 30
        v_margin = 30

        lower_hsv = (
            max(0, int(ranges['h']['min'] - h_margin)),
            max(0, int(ranges['s']['min'] - s_margin)),
            max(0, int(ranges['v']['min'] - v_margin))
        )
        upper_hsv = (
            min(179, int(ranges['h']['max'] + h_margin)),
            min(255, int(ranges['s']['max'] + s_margin)),
            min(255, int(ranges['v']['max'] + v_margin))
        )

        self.lowerHSV = lower_hsv
        self.upperHSV = upper_hsv

        self.lblLower.setText(f"Lower: H {lower_hsv[0]}; S {lower_hsv[1]}; V {lower_hsv[2]}")
        self.lblUpper.setText(f"Upper: H {upper_hsv[0]}; S {upper_hsv[1]}; V {upper_hsv[2]}")

        # Update slider: UPPER
        self.cboxSetMode.setCurrentText("UPPER")
        self.selectedHue = upper_hsv[0] * 2
        self.selectedSaturation = upper_hsv[1]
        self.selectedValue = upper_hsv[2]
        self.sliderH.setValue(self.selectedHue)
        self.sliderS.setValue(self.selectedSaturation)
        self.sliderV.setValue(self.selectedValue)

        # Update slider: LOWER
        self.cboxSetMode.setCurrentText("LOWER")
        self.selectedHue = lower_hsv[0] * 2
        self.selectedSaturation = lower_hsv[1]
        self.selectedValue = lower_hsv[2]
        self.sliderH.setValue(self.selectedHue)
        self.sliderS.setValue(self.selectedSaturation)
        self.sliderV.setValue(self.selectedValue)

        self.updateMask()
        self.updatePreviewHsvSpace()

    # ============== PICK COLOR ==============
    def toggle_color_picker(self):
        self.is_picking_color = not self.is_picking_color
        if self.is_picking_color:
            self.btnPickColor.setText("Stop Picking")
            self.color_picker_timer.start(50)
        else:
            self.btnPickColor.setText("Pick Color")
            self.color_picker_timer.stop()

    def update_color_from_screen(self):
        try:
            x, y = pyautogui.position()
            screenshot = pyautogui.screenshot(region=(x, y, 1, 1))
            color = screenshot.getpixel((0, 0))

            bgr = np.uint8([[list(color)[::-1]]])  # RGB -> BGR
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[0][0]

            self.sliderH.setValue(int(hsv[0] * 2))
            self.sliderS.setValue(int(hsv[1]))
            self.sliderV.setValue(int(hsv[2]))
        except Exception as e:
            print(f"Error picking color: {e}")

    # ============== LAIN-LAIN ==============
    def loadHsvSpace(self):
        """Membuat image BGR yang merepresentasikan rentang HSV (H=0..179, S=0..255, V=255)."""
        h, s = np.meshgrid(np.linspace(0, 179, 180), np.linspace(0, 255, 256))
        v = np.full_like(h, 255)
        hsv = np.stack([h, s, v], axis=2).astype(np.uint8)
        self.imgHsvSpace = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def onBtnCopyClicked(self):
        range_text = (
            f"HSV Ranges:\n"
            f"Upper: {self.upperHSV}\n"
            f"Lower: {self.lowerHSV}\n"
            f"\nFor code:\n"
            f"lower = np.array({self.lowerHSV})\n"
            f"upper = np.array({self.upperHSV})"
        )
        QApplication.clipboard().setText(range_text)
        print(range_text)

    def updatePreviewHsvSpace(self):
        if self.imgHsvSpace is None:
            return

        frame_HSV = cv2.cvtColor(self.imgHsvSpace, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(frame_HSV, np.array(self.lowerHSV), np.array(self.upperHSV))
        masked = cv2.bitwise_and(self.imgHsvSpace, self.imgHsvSpace, mask=mask)

        _asQImage = QImage(
            masked.data,
            masked.shape[1],
            masked.shape[0],
            masked.shape[1]*3,
            QImage.Format_RGB888
        ).rgbSwapped()

        self.previewHsvSpace.setPixmap(
            QPixmap.fromImage(_asQImage).scaled(
                self.previewHsvSpace.width(),
                self.previewHsvSpace.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
        )

    def updateHSVPreview(self):
        prevH = generateSolidColorPixmap(
            80, 80, QColor.fromHsv(self.selectedHue, 255, 255)
        )
        self.previewH.setPixmap(QPixmap.fromImage(prevH))

        prevS = generateSolidColorPixmap(
            80, 80, QColor.fromHsv(self.selectedHue, self.selectedSaturation, 255)
        )
        self.previewS.setPixmap(QPixmap.fromImage(prevS))

        prevV = generateSolidColorPixmap(
            80, 80, QColor.fromHsv(self.selectedHue, self.selectedSaturation, self.selectedValue)
        )
        self.previewV.setPixmap(QPixmap.fromImage(prevV))

        if self.cboxSetMode.currentText() == "UPPER":
            self.upperHSV = (
                self.selectedHue // 2,
                self.selectedSaturation,
                self.selectedValue
            )
            self.lblUpper.setText(
                f"Upper: H {self.upperHSV[0]}; S {self.upperHSV[1]}; V {self.upperHSV[2]}"
            )
        else:
            self.lowerHSV = (
                self.selectedHue // 2,
                self.selectedSaturation,
                self.selectedValue
            )
            self.lblLower.setText(
                f"Lower: H {self.lowerHSV[0]}; S {self.lowerHSV[1]}; V {self.lowerHSV[2]}"
            )

        self.updateMask()
        self.updatePreviewHsvSpace()

    def updateRawImg(self, img):
        if img is None:
            return
        self.imgRaw = img.copy()

        preview_width = self.previewRaw.width()
        preview_height = self.previewRaw.height()

        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        bytes_per_line = img.shape[1] * 3

        _imgAsQImg = QImage(
            rgb_img.data,
            img.shape[1],
            img.shape[0],
            bytes_per_line,
            QImage.Format_RGB888
        )
        pixmap = QPixmap.fromImage(_imgAsQImg)
        scaled_pixmap = pixmap.scaled(
            preview_width,
            preview_height,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.previewRaw.setPixmap(scaled_pixmap)

        # Simpan skala untuk keperluan sampling & real-time zoom
        self.preview_scale = {
            'original_size': (img.shape[1], img.shape[0]),
            'preview_size': (scaled_pixmap.width(), scaled_pixmap.height()),
            'offset_x': (preview_width - scaled_pixmap.width()) // 2,
            'offset_y': (preview_height - scaled_pixmap.height()) // 2
        }

        # Jika ada titik sampling, gambar ulang
        if self.selected_points:
            self.draw_sample_points()

    def updateMask(self):
        if self.imgRaw is None:
            return

        hsv_img = cv2.cvtColor(self.imgRaw, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_img, np.array(self.lowerHSV), np.array(self.upperHSV))

        # Erode
        if self.cboxErode.isChecked():
            kernel_size = self.sliderErotion.value()
            mask = cv2.erode(mask, np.ones((kernel_size, kernel_size), dtype=np.uint8))

        # Dilate
        if self.cboxDilate.isChecked():
            kernel_size = self.sliderDilation.value()
            mask = cv2.dilate(mask, np.ones((kernel_size, kernel_size), dtype=np.uint8))

        self.updateMaskedRaw(mask)

        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        _asQImage = QImage(
            mask_bgr.data,
            mask_bgr.shape[1],
            mask_bgr.shape[0],
            mask_bgr.shape[1]*3,
            QImage.Format_RGB888
        ).rgbSwapped()

        self.previewMask.setPixmap(
            QPixmap.fromImage(_asQImage).scaled(
                self.previewMask.width(),
                self.previewMask.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
        )

        # Pastikan titik sampling tetap terlihat
        if self.selected_points:
            self.draw_sample_points()

    def updateMaskedRaw(self, mask):
        if self.imgRaw is None:
            return

        masked = cv2.bitwise_and(self.imgRaw, self.imgRaw, mask=mask)
        _asQImage = QImage(
            masked.data,
            masked.shape[1],
            masked.shape[0],
            masked.shape[1]*3,
            QImage.Format_RGB888
        ).rgbSwapped()

        self.previewMaskedRaw.setPixmap(
            QPixmap.fromImage(_asQImage).scaled(
                self.previewMaskedRaw.width(),
                self.previewMaskedRaw.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
        )

    # ============== EVENT HANDLER ==============
    def onCBoxModeChanged(self, text):
        if text == "UPPER":
            self.selectedHue = self.upperHSV[0] * 2
            self.selectedSaturation = self.upperHSV[1]
            self.selectedValue = self.upperHSV[2]
        else:
            self.selectedHue = self.lowerHSV[0] * 2
            self.selectedSaturation = self.lowerHSV[1]
            self.selectedValue = self.lowerHSV[2]

        # Blokir sinyal agar tidak memicu updateHSVPreview berkali-kali
        self.sliderH.blockSignals(True)
        self.sliderS.blockSignals(True)
        self.sliderV.blockSignals(True)

        self.sliderH.setValue(self.selectedHue)
        self.sliderS.setValue(self.selectedSaturation)
        self.sliderV.setValue(self.selectedValue)

        self.sliderH.blockSignals(False)
        self.sliderS.blockSignals(False)
        self.sliderV.blockSignals(False)

        self.updateHSVPreview()

    def onHChanged(self):
        val = self.sliderH.value()
        self.lblH.setText(f"H: {val}")
        self.selectedHue = val
        self.updateHSVPreview()

    def onSChanged(self):
        val = self.sliderS.value()
        self.lblS.setText(f"S: {val}")
        self.selectedSaturation = val
        self.updateHSVPreview()

    def onVChanged(self):
        val = self.sliderV.value()
        self.lblV.setText(f"V: {val}")
        self.selectedValue = val
        self.updateHSVPreview()

    def onSliderErodeChanged(self):
        val = self.sliderErotion.value()
        self.cboxErode.setText(f"Erode ({val})")
        self.updateMask()

    def onSliderDilateChanged(self):
        val = self.sliderDilation.value()
        self.cboxDilate.setText(f"Dilate ({val})")
        self.updateMask()

    def onBtnOpenClicked(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image",
            "",
            "All Files (*);;Image Files (*.jpg *.jpeg *.png *.bmp)",
            options=options
        )
        if not fileName:
            return
        img = cv2.imread(fileName)
        if img is not None:
            self.updateRawImg(img)


# ============== MAIN ==============
if __name__ == "__main__":
    app = QApplication(sys.argv)

    widget = ImprovedHSVPicker()

    # Ambil resolusi layar
    screen = QGuiApplication.primaryScreen()
    screen_rect = screen.geometry()
    screen_width, screen_height = screen_rect.width(), screen_rect.height()

    # Buat ukuran hampir full, misal 95% lebar dan 90% tinggi
    target_width = int(0.95 * screen_width)
    target_height = int(0.9 * screen_height)

    widget.setFixedSize(target_width, target_height)

    # Letakkan di tengah
    pos_x = (screen_width - target_width) // 2
    pos_y = (screen_height - target_height) // 2
    widget.move(pos_x, pos_y)

    widget.show()
    sys.exit(app.exec_())
