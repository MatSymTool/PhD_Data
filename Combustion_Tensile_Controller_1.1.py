import os
import re
import sys
import time
import csv
import queue
import threading
import logging
from datetime import datetime
import numpy as np
from serial import Serial
from serial.tools import list_ports
import pyqtgraph as pg
from PyQt6.QtGui import QFont
import serial

import matplotlib
matplotlib.use("QtAgg")

# PyQt6 imports
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QLineEdit, QComboBox, QPushButton, QGroupBox, QMessageBox, QFileDialog,
    QScrollArea, QSpinBox, QDialog, QDoubleSpinBox, QTabWidget, QSplitter
)
from PyQt6.QtCore import QTimer, QThread, pyqtSignal, Qt, QObject

# Add these imports at the top
from queue import Queue
from dataclasses import dataclass
from typing import Optional

@dataclass
class SerialCommand:
    cmd: str
    response_needed: bool = False

class  DeviceSerialWorker(QObject):
    """Handles serial communication in a background thread."""
    data_received = pyqtSignal(str)  # Emits received data
    error = pyqtSignal(str)          # Emits errors
    
    def __init__(self, port: str, baudrate: int, timeout: float = 0.1):
        super().__init__()
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.serial = None
        self.running = False
        self.cmd_queue = Queue()
        
    def run(self):
        try:
            self.serial = Serial(self.port, self.baudrate, timeout=self.timeout)
            self.running = True
            
            while self.running:
                # Handle outgoing commands
                try:
                    while not self.cmd_queue.empty():
                        cmd = self.cmd_queue.get_nowait()
                        if isinstance(cmd, SerialCommand):
                            self.serial.write(f"{cmd.cmd}\n".encode('ascii'))
                            if cmd.response_needed:
                                resp = self.serial.readline().decode('ascii').strip()
                                self.data_received.emit(resp)
                except Exception as e:
                    self.error.emit(f"Command error: {e}")
                
                # Handle incoming data
                if self.serial.in_waiting:
                    try:
                        data = self.serial.readline().decode('ascii').strip()
                        if data:
                            self.data_received.emit(data)
                    except Exception as e:
                        self.error.emit(f"Read error: {e}")
                
                QThread.msleep(10)  # Don't hog CPU
                
        except Exception as e:
            self.error.emit(f"Serial worker error: {e}")
        finally:
            if self.serial:
                self.serial.close()

    def stop(self):
        self.running = False

    def send_command(self, cmd: str, need_response: bool = False):
        """Queue a command to be sent to the device."""
        self.cmd_queue.put(SerialCommand(cmd, need_response))

# -----------------------
# Setup logging
# -----------------------
log_directory = os.path.dirname(os.path.abspath(__file__))
logs_dir = os.path.join(log_directory, "logs")
os.makedirs(logs_dir, exist_ok=True)
log_filename = os.path.join(logs_dir, f"Combustion_Controller_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
print(log_filename)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler(log_filename)
file_handler.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter("(%(asctime)s) - %(levelname)s - %(message)s", datefmt="%H:%M:%S")
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
else:
    # Avoid duplicate handlers on hot-reload
    logger.handlers.clear()
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

serial_lock = threading.Lock()

# -----------------------
# Global variables (for controller channels)
# -----------------------
unit_id_widgets = []
baud_rate_widgets = []
com_channel_widgets = []
flow_rate_widgets = []
status_indicators = []  # For connection status circles
serial_connections = [None] * 3
connect_status = [False] * 3
gas_types = ["H2", "Air", "N2"]
readings_dict = {gas: {} for gas in gas_types}
data_queue = queue.Queue()

# -----------------------
# Thread to handle serial connection (non-blocking UI)
# -----------------------
class SerialConnectThread(QThread):
    finished = pyqtSignal(dict)  # Will emit result dict

    def __init__(self, index):
        super().__init__()
        self.index = index

    def run(self):
        result = {}
        try:
            sel_baud_rate = int(baud_rate_widgets[self.index].text())
            com_text = com_channel_widgets[self.index].currentText()
            if not com_text:
                result["error"] = "Please select a COM port."
            else:
                com_port = com_text.split(' - ')[0]
                unit_id = unit_id_widgets[self.index].text().strip()
                serial_conn = Serial(port=com_port, baudrate=sel_baud_rate, timeout=0.1)
                serial_conn.reset_input_buffer()
                serial_conn.write(f"{unit_id}DV 16\r".encode('ascii'))
                response = serial_conn.readline().decode('ascii').strip().split()
                if len(response) < 2:
                    result["error"] = "No valid response from controller."
                    serial_conn.close()
                else:
                    result["com_port"] = com_port
                    result["serial_conn"] = serial_conn
                    result["stored_gas"] = response[1]
        except Exception as e:
            result["error"] = str(e)
        self.finished.emit(result)

# -----------------------
# Helper function to send an ASCII command over serial
# -----------------------
def send_ascii_command(serial_conn, command):
    if serial_conn and serial_conn.is_open:
        with serial_lock:
            serial_conn.write(f"{command}\r".encode('ascii'))
            response = serial_conn.readline().decode('ascii').strip()
        return response
    return None

# -----------------------
# Serial read thread
# -----------------------
def serial_read_thread():
    while True:
        for index, gas in enumerate(gas_types):
            if connect_status[index]:
                unit_id = unit_id_widgets[index].text().strip()
                consolidated_command = f"{unit_id}DV 7\r"
                retries = 0
                max_retries = 2
                while retries <= max_retries:
                    try:
                        response = send_ascii_command(serial_connections[index], consolidated_command)
                    except Exception:
                        break

                    if response:
                        parts = response.split()
                        try:
                            if len(parts) < 4:
                                raise IndexError("Not enough data")
                            flow_rate = parts[1].lstrip('+')
                            try:
                                if float(flow_rate) < 0:
                                    flow_rate = "0"
                            except ValueError:
                                pass
                            temperature = parts[3].lstrip('+')
                            setpoint = parts[2].lstrip('+')
                            controller_data = {
                                "Flow rate [SLPM]": flow_rate,
                                "Temperature [°C]": temperature,
                                "Setpoint [SLPM]": setpoint
                            }
                            data_queue.put((index, controller_data))
                            retries = 0
                            break
                        except IndexError:
                            logger.warning(f"Parsing error for {gas}. Retrying... ({retries+1}/{max_retries+1})")
                            retries += 1
                            if retries > max_retries:
                                logger.error(f"Failed parsing response from {gas} after {max_retries} retries.")
                                break
                    else:
                        break
        time.sleep(0.05)

class TensileForceReader(QObject):
    force_updated = pyqtSignal(float)
    error = pyqtSignal(str)
    
    def __init__(self, conn, update_interval_ms=100):
        super().__init__()
        self.conn = conn
        self.update_interval = update_interval_ms / 1000.0
        self.running = False
        
    def run(self):
        self.running = True
        while self.running:
            try:
                with serial_lock:
                    if self.conn and self.conn.in_waiting:
                        line = self.conn.readline().decode('utf-8', errors='ignore').strip()
                        if line:
                            m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", line)
                            if m:
                                force_val = float(m.group(0))
                                self.force_updated.emit(force_val)
                time.sleep(self.update_interval)
            except Exception as e:
                self.error.emit(f"Force read error: {e}")
                
    def stop(self):
        self.running = False

# -----------------------
# Temperature widget (refactored from QDialog -> QWidget for embedding)
# -----------------------
class TemperatureWidget(QWidget):
    temperature_updated = pyqtSignal(float)
    def __init__(self, parent=None):
        super().__init__(parent)

        # Temperature sensor variables
        self.temp_ser = None
        self.temp_time_data = []
        self.temp_values = []
        self.temp_recording = False
        self.temp_record_start_time = None
        self.temp_csv_file = None
        self.temp_csv_writer = None
        self.temp_csv_data = []  # Buffer to store CSV rows during recording
        self.temp_val = None
        self.temp_baud_rate = 115200

        # ----- Top bar: COM + baud + connect + log file -----
        top_layout = QHBoxLayout()
        top_layout.addWidget(QLabel("Temp COM Port:"))
        self.temp_combobox = QComboBox()
        top_layout.addWidget(self.temp_combobox)
        #self.temp_combobox.setFixedWidth(200)
        connect_btn = QPushButton("Connect")
        connect_btn.clicked.connect(self.connect_temp_sensor)
        top_layout.addWidget(connect_btn)
        disconnect_btn = QPushButton("Disconnect")
        disconnect_btn.clicked.connect(self.disconnect_temp_sensor)
        top_layout.addWidget(disconnect_btn)
        log_btn = QPushButton("Set Log File")
        log_btn.clicked.connect(self.select_csv_file)
        top_layout.addWidget(log_btn)

        # Main layout
        self.layout_main = QVBoxLayout(self)
        self.layout_main.addLayout(top_layout)

        # Timer to update COM ports
        self.temp_com_timer = QTimer(self)
        self.temp_com_timer.timeout.connect(self.update_temp_com_ports)
        self.temp_com_timer.start(1000)

        # Current temperature label
        self.current_temp_label = QLabel("Current Temp: - °C")
        self.current_temp_label.setStyleSheet("font-family: Courier New; font-size: 12pt; font-weight: bold;")
        self.layout_main.addWidget(self.current_temp_label)

        # Control layout with record button + info
        control_layout = QHBoxLayout()
        self.record_btn = QPushButton("Start Recording")
        self.record_btn.clicked.connect(self.toggle_temp_record)
        control_layout.addWidget(self.record_btn)
        self.info_label = QLabel("No Data")
        self.info_label.setStyleSheet("font-family: Courier New; font-size: 10pt;")
        control_layout.addWidget(self.info_label)
        control_layout.addStretch()
        self.layout_main.addLayout(control_layout)

        # Timer layout (hh:mm)
        timer_layout = QHBoxLayout()
        timer_layout.addWidget(QLabel("Timer: "))
        self.timer_spin = QSpinBox(); self.timer_spin.setRange(0, 24); self.timer_spin.setValue(0)
        timer_layout.addWidget(self.timer_spin)
        timer_layout.addWidget(QLabel("h"))
        self.min_spin = QSpinBox(); self.min_spin.setRange(0, 59); self.min_spin.setValue(0)
        timer_layout.addWidget(self.min_spin)
        timer_layout.addWidget(QLabel("min"))
        self.start_btn = QPushButton("Start Timer")
        self.start_btn.clicked.connect(self.start_timer)
        timer_layout.addWidget(self.start_btn)
        self.timer_label = QLabel("Time left: 00:00:00")
        timer_layout.addWidget(self.timer_label)
        timer_layout.addStretch()
        self.layout_main.addLayout(timer_layout)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_timer)
        self.time_left = 0
        self._timer_running = False

        # Plot
        styles = {"color": "black", "font-size": "18px"}
        my_font = QFont(); my_font.setPixelSize(18)
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('w')
        self.plot_widget.showGrid(x=True, y=True)
        self.plot_widget.enableAutoRange('xy', True)
        self.plot_widget.setLabel("left", "Temperature (°C)", **styles)
        self.plot_widget.setLabel("bottom", "Time (s)", **styles)
        self.plot_widget.getAxis("bottom").setTickFont(my_font)
        self.plot_widget.getAxis('left').setTickFont(my_font)
        self.plot_widget.getAxis("bottom").setTextPen(pg.mkPen('k'))
        self.plot_widget.getAxis("left").setTextPen(pg.mkPen('k'))
        self.plot_curve = self.plot_widget.plot([], [], pen=pg.mkPen('b', width=2))
        # horizontal lines for min and max temperature
        self.min_temp_line = pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen('r', width=2, style=pg.QtCore.Qt.PenStyle.DashLine))
        self.max_temp_line = pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen('r', width=2, style=pg.QtCore.Qt.PenStyle.DashLine))
        self.plot_widget.addItem(self.min_temp_line)
        self.plot_widget.addItem(self.max_temp_line)
        self.min_temp_line.hide(); self.max_temp_line.hide()
        self.layout_main.addWidget(self.plot_widget)

        # Timer for data acquisition (and live plot update)
        self.temp_timer = QTimer(self)
        self.temp_timer.timeout.connect(self.acquire_temp_data)
        self.temp_timer.start(1000)

    # ----- Temperature widget helpers -----
    def update_temp_com_ports(self):
        ports = list_ports.comports()
        com_ports = sorted([f"{port.device} - {port.description}" for port in ports])
        if not self.temp_combobox.view().isVisible():
            current = self.temp_combobox.currentText()
            self.temp_combobox.blockSignals(True)
            self.temp_combobox.clear()
            self.temp_combobox.addItem("")
            self.temp_combobox.addItems(com_ports)
            if current in com_ports:
                self.temp_combobox.setCurrentText(current)
            else:
                self.temp_combobox.setCurrentIndex(0)
            self.temp_combobox.blockSignals(False)

    def connect_temp_sensor(self):
        selected = self.temp_combobox.currentText()
        if selected:
            com_port = selected.split(" - ")[0]
            try:
                self.temp_ser = Serial(com_port, self.temp_baud_rate, timeout=0.1)
                QMessageBox.information(self, "Connection", f"Connected to temperature sensor on {com_port}.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to connect: {e}")
        else:
            QMessageBox.warning(self, "Warning", "Please select a COM port.")
        logger.info("Temperature controller connected")

    def disconnect_temp_sensor(self):
        selected = self.temp_combobox.currentText()
        if not selected:
            QMessageBox.warning(self, "Warning", "Please select a COM port.")
            return

        com_port = selected.split(" - ")[0]
        
        try:
            conn = None
            if hasattr(self, "temp_ser") and self.temp_ser and getattr(self.temp_ser, "is_open", False) and getattr(self.temp_ser, "port", None) == com_port:
                conn = self.temp_ser

            if conn is None:
                QMessageBox.critical(self, "Error", "No open serial connection for selected COM port.")
                return
        except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to disconnect: {e}")

        with serial_lock:
            try:
                conn.close()
            except Exception:
                pass
            # remove reference so acquire_temp_data won't touch a closed handle
            try:
                self.temp_ser = None
            except Exception:
                pass
        logger.info("Temperature controller disconnected")


    def select_csv_file(self):
        csv_file_path, _ = QFileDialog.getSaveFileName(self, "Select Log File", "", "CSV files (*.csv)")
        if csv_file_path:
            self.temp_csv_file = open(csv_file_path, 'w', newline='')
            self.temp_csv_writer = csv.writer(self.temp_csv_file)
            self.temp_csv_writer.writerow(["Time [s]", "Temperature [°C]"])
            QMessageBox.information(self, "CSV", f"CSV file selected: {csv_file_path}")

    def toggle_temp_record(self):
        if not self.temp_recording:
            if self.temp_csv_file:
                if self.temp_time_data:
                    reply = QMessageBox.question(
                        self, "Confirmation", "Erase all data and start recording?",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                    )
                    if reply == QMessageBox.StandardButton.Yes:
                        self.temp_time_data.clear(); self.temp_values.clear(); self.temp_csv_data.clear()
                        self.temp_csv_file.seek(0); self.temp_csv_file.truncate()
                        self.temp_csv_writer.writerow(["Time [s]", "Temperature [°C]"])
                    else:
                        return
            else:
                reply = QMessageBox.question(
                    self, "Warning", "No CSV file selected. Continue?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                if reply != QMessageBox.StandardButton.Yes:
                    return
            self.temp_record_start_time = time.time()
            self.temp_recording = True
            self.record_btn.setText("Stop Recording")
            self.temp_time_data.clear(); self.temp_values.clear(); self.temp_csv_data.clear()
        else:
            reply = QMessageBox.question(
                self, "Confirmation", "Stop recording?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.temp_recording = False
                self.record_btn.setText("Start Recording")
                if self.temp_csv_file and self.temp_csv_data:
                    for row in self.temp_csv_data:
                        self.temp_csv_writer.writerow(row)
                    self.temp_csv_file.flush(); self.temp_csv_data.clear()

    def start_timer(self):
        if self._timer_running:
            reply = QMessageBox.question(self, "Warning", "Stop Timer?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Yes:
                self.timer.stop(); self._timer_running = False
                self.start_btn.setText("Start Timer")
        else:
            hours = self.timer_spin.value(); minutes = self.min_spin.value()
            self.time_left = hours * 3600 + minutes * 60
            if self.time_left <= 0:
                QMessageBox.warning(self, "Invalid Duration", "Please set a valid timer duration.")
                return
            self.start_btn.setText("Stop Timer")
            self.timer.start(1000); self._timer_running = True

    def update_timer(self):
        self.time_left -= 1
        hours, remainder = divmod(self.time_left, 3600)
        minutes, seconds = divmod(remainder, 60)
        self.timer_label.setText(f"Time left: {hours:02}:{minutes:02}:{seconds:02}")
        if self.time_left <= 0:
            self.timer.stop()
            try:
                win = self.window(); win.raise_(); win.activateWindow(); QApplication.alert(win, 0)
            except Exception:
                pass
            QMessageBox.information(self, "Timer", "Timer is done!")
            QApplication.beep(); self._timer_running = False
            self.start_btn.setText("Start Timer")

    def format_time(self, seconds):
        if seconds < 60:
            return f"{seconds:.0f} s"
        elif seconds < 3600:
            return f"{seconds/60:.1f} min"
        else:
            return f"{seconds/3600:.1f} h"

    def acquire_temp_data(self):
        selected = self.temp_combobox.currentText()
        if not selected:
            return
        com_port = selected.split(" - ")[0]
        try:
            # proceed only if temp_ser exists, is open and matches the selected port
            if not (hasattr(self, "temp_ser") and self.temp_ser and getattr(self.temp_ser, "is_open", False) and getattr(self.temp_ser, "port", None) == com_port):
                # no valid/open temp serial for selected port -> nothing to read
                return
            try:
                if self.temp_ser and self.temp_ser.in_waiting:
                    try:
                        latest_line = None
                        while self.temp_ser.in_waiting:
                            latest_line = self.temp_ser.readline().decode('utf-8').strip()
                        if latest_line:
                            try:
                                temp_val = float(latest_line)
                            except Exception:
                                temp_val = None
                            if temp_val is not None:
                                self.temp_val = temp_val
                                self.current_temp_label.setText(f"Current Temp: {temp_val:.2f} °C")
                                self.temperature_updated.emit(temp_val)  # Emit the signal with the current temperature
                                if self.temp_recording:
                                    if self.temp_record_start_time is None:
                                        self.temp_record_start_time = time.time()
                                    elapsed = time.time() - self.temp_record_start_time
                                    self.temp_time_data.append(elapsed)
                                    self.temp_values.append(temp_val)
                                    self.temp_csv_data.append([datetime.now().strftime("%H:%M:%S"), temp_val])
                                    rec_time = elapsed
                                    avg_temp = f"{np.mean(self.temp_values):.2f} °C"
                                    max_temp = f"{max(self.temp_values):.2f} °C"
                                    labels = ["Recording Time:", "Average Temp:", "Max Temp:"]
                                    values = [self.format_time(rec_time), avg_temp, max_temp]
                                    max_label_len = max(len(lbl) for lbl in labels)
                                    info_text = "\n".join(f"{lbl.ljust(max_label_len)}  {val}" for lbl, val in zip(labels, values))
                                    self.info_label.setText(info_text)
                                    # x-axis label/time scale
                                    if max(self.temp_time_data) < 60:
                                        factor = 1; xlabel = "Time (s)"
                                    elif max(self.temp_time_data) < 3600:
                                        factor = 1/60; xlabel = "Time (min)"
                                    else:
                                        factor = 1/3600; xlabel = "Time (h)"
                                    self.plot_widget.setLabel("bottom", xlabel, **{"color": "black", "font-size": "18px"})
                                    temp_data = [factor * t for t in self.temp_time_data]
                                    self.plot_curve.setData(temp_data, self.temp_values)
                    except Exception as e:
                        logger.error(f"Serial read error (temp): {e}")
            except serial.SerialException as e:
                            # serial port became invalid/closed mid-read — clear reference and log
                            logger.warning(f"Temperature serial error, disconnecting port: {e}")
                            with serial_lock:
                                try:
                                    if hasattr(self, "temp_ser") and self.temp_ser:
                                        try:
                                            self.temp_ser.close()
                                        except Exception:
                                            pass
                                    self.temp_ser = None
                                except Exception:
                                    pass
        except Exception as e:
            logger.error(f"Serial read error (temp): {e}")

    def update_temp_limits(self, min_temp, max_temp):
        try:
            min_val = float(min_temp); max_val = float(max_temp)
            self.min_temp_line.setPos(min_val); self.max_temp_line.setPos(max_val)
            self.min_temp_line.show(); self.max_temp_line.show()
        except (ValueError, TypeError):
            self.min_temp_line.hide(); self.max_temp_line.hide()

    def cleanup(self):
        # Call on app exit if needed
        try:
            if self.temp_csv_file and self.temp_csv_data:
                for row in self.temp_csv_data:
                    self.temp_csv_writer.writerow(row)
                self.temp_csv_file.flush(); self.temp_csv_file.close()
        except Exception:
            pass
        try:
            if self.temp_ser and self.temp_ser.is_open:
                self.temp_ser.close()
        except Exception:
            pass
    

# -----------------------
# Main Window (now with tabs)
# -----------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.saved_motor_speed = None  # Variable to store the new motor speed while paused
        self.tensile_paused = False
        self.setWindowTitle("H Combustion Rig Control Panel 2.0 - by Philipp Schulz")

        # Root widget: QTabWidget
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        #===========================================
        # Tab 1: Unified (gas flows + temperature)
        #===========================================
        self.tab_unified = QWidget()
        self.tabs.addTab(self.tab_unified, "Combustion")

        # Build unified tab with a vertical splitter
        unified_layout = QVBoxLayout(self.tab_unified)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        unified_layout.addWidget(splitter)

        # Top of splitter: gas control panel inside a scroll area
        controls_container = QWidget()
        self.content_layout = QVBoxLayout(controls_container)
        self._build_control_panel_ui()  # populates self.content_layout

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(controls_container)
        scroll_area.setMinimumWidth(120)
        splitter.addWidget(scroll_area)

        # Bottom of splitter: temperature widget
        self.temp_widget = TemperatureWidget(self)

        self.temp_widget.temperature_updated.connect(self.update_sample_temp_label)
        self.temp_widget.setMinimumWidth(120)
        splitter.addWidget(self.temp_widget)
        # Make both panes collapsible and set stretch factors so the left pane takes most space by default.
        splitter.setCollapsible(0, True)
        splitter.setCollapsible(1, True)
        splitter.setStretchFactor(0, 3)  # controls expand/shrink priority
        splitter.setStretchFactor(1, 1)
        # initial sizes favour the left panel (plot can be made <50% by dragging)
        splitter.setSizes([700, 500])



        # Timers for UI updates and COM port updates
        self.ui_timer = QTimer(); self.ui_timer.timeout.connect(self.update_ui); self.ui_timer.start(50)
        self.tensile_timer = QTimer(); self.tensile_timer.timeout.connect(self.update_tensile_data)  # Unified function to update readings and generate data
        #self.tensile_timer.start(50)  # Adjust the interval as needed
        self.com_timer = QTimer(); self.com_timer.timeout.connect(self.update_com_channels); self.com_timer.start(1000)
        # Start serial read thread
        threading.Thread(target=serial_read_thread, daemon=True).start()

        # Alarms & cruise control
        self.temp_alarm_timer = QTimer(); self.temp_alarm_timer.timeout.connect(self.check_temperature); self.temp_alarm_active = False
        self.flow_alarm_timer = QTimer(); self.flow_alarm_timer.timeout.connect(self.check_flows); self.flow_alarm_active = False
        self.cruise_control_timer = QTimer(); self.cruise_control_timer.timeout.connect(self.control_cruise); self.cruise_control_active = False
        self._flow_warn_times = {}  # index -> last below-threshold warning timestamp (3 s cooldown)

        # Periodic status update every 5 minutes
        self.status_update_timer = QTimer()
        self.status_update_timer.timeout.connect(self.log_status_update)
        self.status_update_timer.start(10 * 60 * 1000)

        # Resize to a sensible default
        screen = QApplication.primaryScreen().availableGeometry()
        self.setGeometry(50, 30, int(screen.width()*0.9), int(screen.height()*0.9))
        
        #===========================================
        # Tab 2: Tensile Stage
        #===========================================
        content_widget = QWidget()
        content_widget.setObjectName("tensile_content_widget")
        top_layout_2 = QVBoxLayout()  # existing code uses top_layout_2 extensively
        content_layout = QVBoxLayout(content_widget)
        content_layout.addLayout(top_layout_2)

        # Wrap the content widget in a scroll area and add that to the tab widget
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content_widget)
        self.tabs.addTab(scroll, "Tensile Stage")

        # keep a reference in case other code expects self.tensile_layout
        self.tensile_layout = content_layout
        #------------------------
        # COM connection

        ard_con_box = QGroupBox("Tensile Stage Readings")
        ard_con_box.setStyleSheet("font-size: 12pt;")
        ard_con_layout = QHBoxLayout()

        ard_con_layout.addWidget(QLabel("Arduino COM Port:"))
        self.tensile_combobox = QComboBox()
        self.tensile_combobox.setFixedWidth(200)
        ard_con_layout.addWidget(self.tensile_combobox)

        ard_connect_btn = QPushButton("Connect")
        ard_connect_btn.clicked.connect(self.connect_tensile_arduino)
        ard_connect_btn.setFixedWidth(150)
        ard_con_layout.addWidget(ard_connect_btn)

        ard_disconnect_btn = QPushButton("Disconnect")
        ard_disconnect_btn.clicked.connect(self.disconnect_tensile_arduino)
        ard_disconnect_btn.setFixedWidth(150)
        ard_con_layout.addWidget(ard_disconnect_btn)

        log_btn = QPushButton("Set Log File")
        log_btn.clicked.connect(self.select_csv_file)
        log_btn.setFixedWidth(150)
        ard_con_layout.addWidget(log_btn)
        
        exp_btn = QPushButton("Export Tensile Data")
        exp_btn.clicked.connect(self.write_tensile_data)
        exp_btn.setFixedWidth(150)
        ard_con_layout.addWidget(exp_btn)

        ard_con_layout.addStretch()

        ard_con_box.setLayout(ard_con_layout)
        top_layout_2.addWidget(ard_con_box)

        #------------------------
        # Tensile stage readings
        
        readings_box = QGroupBox("Tensile Stage Readings")
        readings_box.setStyleSheet("font-size: 12pt;")
        readings_inner_layout = QHBoxLayout()
        
        
        mode_layout = QHBoxLayout()
        mode_lbl = QLabel("Mode:")
        self.mode_label = QLabel("Tensile"); self.mode_label.setFrameStyle(QLabel.Shape.Box | QLabel.Shadow.Sunken)
        mode_layout.addWidget(mode_lbl)
        mode_layout.addWidget(self.mode_label)
        mode_layout.addStretch()
        readings_inner_layout.addLayout(mode_layout)

        elapsed_time_layout = QHBoxLayout()
        time_lbl = QLabel("Elapsed:")
        elapsed_time_layout.addWidget(time_lbl)
        self.time_lbl = QLabel("0"); self.time_lbl.setFrameStyle(QLabel.Shape.Box | QLabel.Shadow.Sunken);self.time_lbl.setFixedWidth(60)        
        elapsed_time_layout.addWidget(self.time_lbl)
        self.time_unit_lbl = QLabel("s")
        elapsed_time_layout.addWidget(self.time_unit_lbl)
        elapsed_time_layout.addStretch()
        readings_inner_layout.addLayout(elapsed_time_layout)

        force_layout = QHBoxLayout()
        mode_lbl = QLabel("Force:")
        self.force_label = QLabel("0"); self.force_label.setFrameStyle(QLabel.Shape.Box | QLabel.Shadow.Sunken);self.force_label.setFixedWidth(60)
        force_layout.addWidget(mode_lbl)
        force_layout.addWidget(self.force_label)
        unit_lbl = QLabel("N")
        force_layout.addWidget(unit_lbl)
        force_layout.addStretch()
        readings_inner_layout.addLayout(force_layout)

        ext_layout = QHBoxLayout()
        mode_lbl = QLabel("Elongation:")
        ext_layout.addWidget(mode_lbl)
        self.ext_label = QLabel("0"); self.ext_label.setFrameStyle(QLabel.Shape.Box | QLabel.Shadow.Sunken);self.ext_label.setFixedWidth(80)
        ext_layout.addWidget(self.ext_label)
        unit_lbl = QLabel("mm")
        ext_layout.addWidget(unit_lbl)
        ext_layout.addStretch()
        readings_inner_layout.addLayout(ext_layout)

        sample_temp_layout = QHBoxLayout()
        mode_lbl = QLabel("Temp. Sample:")
        self.sample_temp_label = QLabel("0"); self.sample_temp_label.setFrameStyle(QLabel.Shape.Box | QLabel.Shadow.Sunken);self.sample_temp_label.setFixedWidth(60)
        sample_temp_layout.addWidget(mode_lbl)
        sample_temp_layout.addWidget(self.sample_temp_label)
        unit_lbl = QLabel("°C")
        sample_temp_layout.addWidget(unit_lbl)
        sample_temp_layout.addStretch()
        readings_inner_layout.addLayout(sample_temp_layout)

        loadcell_temp_layout = QHBoxLayout()
        mode_lbl = QLabel("Temp. Load Cell:")
        self.loadcell_temp_label = QLabel("0"); self.loadcell_temp_label.setFrameStyle(QLabel.Shape.Box | QLabel.Shadow.Sunken);self.loadcell_temp_label.setFixedWidth(60)
        loadcell_temp_layout.addWidget(mode_lbl)
        loadcell_temp_layout.addWidget(self.loadcell_temp_label)
        unit_lbl = QLabel("°C")
        loadcell_temp_layout.addWidget(unit_lbl)
        loadcell_temp_layout.addStretch()
        readings_inner_layout.addLayout(loadcell_temp_layout)

        speed_layout = QHBoxLayout()
        mode_lbl = QLabel("Motor Speed:")
        self.speed_label = QLabel("0"); self.speed_label.setFrameStyle(QLabel.Shape.Box | QLabel.Shadow.Sunken);self.speed_label.setFixedWidth(60)
        speed_layout.addWidget(mode_lbl)
        speed_layout.addWidget(self.speed_label)
        unit_lbl = QLabel("mm/min")
        speed_layout.addWidget(unit_lbl)
        speed_layout.addStretch()
        readings_inner_layout.addLayout(speed_layout)

        #readings_inner_layout.addStretch()

        readings_box.setLayout(readings_inner_layout)
        top_layout_2.addWidget(readings_box)
        
        #------------------------
        # Plot window 
        
        plot_box = QWidget()
        plot_box.setStyleSheet("font-size: 12pt;")
        plot_inner_layout = QHBoxLayout()
        
        styles = {"color": "black", "font-size": "18px"}
        my_font = QFont(); my_font.setPixelSize(18)
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('w')
        self.plot_widget.showGrid(x=True, y=True)
        self.plot_widget.enableAutoRange('xy', True)
        self.plot_widget.setLabel("left", "Force (N)", **styles)
        self.plot_widget.setLabel("bottom", "Time (s)", **styles)
        self.plot_widget.getAxis("bottom").setTickFont(my_font)
        self.plot_widget.getAxis('left').setTickFont(my_font)
        self.plot_widget.getAxis("bottom").setTextPen(pg.mkPen('k'))
        self.plot_widget.getAxis("left").setTextPen(pg.mkPen('k'))
        self.plot_tens_curve = self.plot_widget.plot([], [], pen=pg.mkPen('b', width=2))

        self.tensile_time_data = []
        self.tensile_force_values = []
        self.speed_array = []
        self.sample_temp_array = []
        self.loadcell_temp_array =[]
        self.elongation_array = []
        self.elongation_array_ind = []
        self.tensile_stress_values = []
        self.tensile_start_time = None
        self.tensile_worker = None
        self.tensile_baud_rate = 115200
        self.micro = 10000
        self.start_speed = 0
        self.jaw_ctrl_var = 0
        self.open_jaw_var = 0
        self.close_jaw_var = 0
        self.force_val_raw = 0
        self.force_val = 0
        self.zero_F = 0
        self.elon = 0
        self.zero_elon = 0        
        self.screwjack_speed = 0.25 #mm/rev
        self.tensile_dir = 1    #spinning direction for tension (1=CW, 0=CCW)
        
        # ======== DEFAULT STARTUP VALUES ========
        self.tensile_update_interval = 100 #ms -> time interval for updating tensile data
        self.gear_ratio = "50:1" #either None or 50:1
        default_loadcell = "small" #Change default load cell here (either "small" or "big")
        
        #Do not change the following values to switch between big and small load cell
        self.loadcell = "2.5 kN" if default_loadcell == "small" else "50 kN"
        self.calibration_factor = 2715.0 if default_loadcell == "small" else 0 #TODO: update big cell factor
        self.force_lim = 2000 if default_loadcell == "small" else 45000
        # =========================================
        
        self.tensile_start_date = datetime.today().strftime('%Y-%m-%d')

        plot_inner_layout.addWidget(self.plot_widget)
        
            #------------------------
            # Controls window        
            #------------------------

        tensile_controls_layout = QVBoxLayout()
        container_widget = QWidget()
        container_widget.setLayout(tensile_controls_layout)
        container_widget.setStyleSheet("font-size: 12pt;") 
        container_widget.setFixedWidth(300)
        
        #Stage Setup
        self.stage_setup_btn = QPushButton("Tensile Stage Setup")
        self.stage_setup_btn.clicked.connect(self.stage_setup)
        tensile_controls_layout.addWidget(self.stage_setup_btn)
        
        #Initial Setup
        initial_setup_box = QGroupBox("Initial Setup")
        initial_setup_box.setStyleSheet("font-size: 12pt;")
        initial_setup_layout = QGridLayout()

        initial_setup_layout.addWidget(QLabel("Motor Speed\n[mm/min]"), 1, 0)
        self.MMotor_speed = QLineEdit()
        self.MMotor_speed.setText("0.100")
        initial_setup_layout.addWidget(self.MMotor_speed, 1, 1)
        self.btn_apply_speed = QPushButton("Apply")
        self.btn_apply_speed.clicked.connect(self.apply__motor_speed)
        self.btn_apply_speed.setEnabled(False) 
        initial_setup_layout.addWidget(self.btn_apply_speed,1,2)

        self.btn_zero_F = QPushButton("Zero Force")
        self.btn_zero_F.clicked.connect(self.zero_forces)
        initial_setup_layout.addWidget(self.btn_zero_F, 2, 0, 1,3)
        
        self.btn_zero_elon = QPushButton("Zero Elongation")
        self.btn_zero_elon.clicked.connect(self.zero_elongation)
        initial_setup_layout.addWidget(self.btn_zero_elon, 3, 0, 1, 3)

        initial_setup_box.setLayout(initial_setup_layout)
        tensile_controls_layout.addWidget(initial_setup_box)
        
        #Start/Stop Controls    
        movement_ctrl_box = QGroupBox("Start/ Stop Control")
        movement_ctrl_box.setStyleSheet("font-size: 12pt;")
        movement_ctrl_layout = QGridLayout()
        
        self.btn_start_tens = QPushButton("Start Test")
        self.btn_start_tens.clicked.connect(self.start_tensile_test)
        movement_ctrl_layout.addWidget(self.btn_start_tens, 0, 0)

        self.btn_pause_tens = QPushButton("Pause")
        self.btn_pause_tens.clicked.connect(self.pause_tensile_test)
        movement_ctrl_layout.addWidget(self.btn_pause_tens, 1, 0)
        self.btn_pause_tens.setEnabled(False) 

        self.btn_stop_tens = QPushButton("Stop Test")
        self.btn_stop_tens.clicked.connect(self.stop_tensile_test)
        movement_ctrl_layout.addWidget(self.btn_stop_tens, 2, 0)
        self.btn_stop_tens.setEnabled(False) 

        movement_ctrl_box.setLayout(movement_ctrl_layout)
        tensile_controls_layout.addWidget(movement_ctrl_box)
        
        #Start/Stop Controls    
        jaw_ctrl_box = QGroupBox("Jaw Control")
        jaw_ctrl_box.setStyleSheet("font-size: 12pt;")
        jaw_ctrl_layout = QGridLayout()
    
        self.btn_open_jaws = QPushButton("Open Jaws")
        self.btn_open_jaws.clicked.connect(self.open_jaws)
        jaw_ctrl_layout.addWidget(self.btn_open_jaws, 0, 0)

        self.btn_close_jaws = QPushButton("Close Jaws")
        self.btn_close_jaws.clicked.connect(self.close_jaws)
        jaw_ctrl_layout.addWidget(self.btn_close_jaws, 1, 0)

        jaw_ctrl_box.setLayout(jaw_ctrl_layout)
        tensile_controls_layout.addWidget(jaw_ctrl_box)


        #Plot Controls    
        plot_ctrl_box = QGroupBox("Plot Setup")
        plot_ctrl_box.setStyleSheet("font-size: 12pt;")
        plot_ctrl_layout = QGridLayout()

        plot_ctrl_layout.addWidget(QLabel("Sample Thickness\n[mm]"), 0, 0)
        self.sample_thickness = QLineEdit()
        self.sample_thickness.setText("1.00")
        plot_ctrl_layout.addWidget(self.sample_thickness, 0, 1)

        plot_ctrl_layout.addWidget(QLabel("Gauge Length [mm]"), 1, 0)
        self.sample_gauge = QLineEdit()
        self.sample_gauge.setText("1.00")
        plot_ctrl_layout.addWidget(self.sample_gauge, 1, 1)

        plot_ctrl_layout.addWidget(QLabel("x-Axis Unit"), 2, 0)
        self.xaxis_box = QComboBox()
        self.xaxis_box.addItems(["Time", "Elongation"])
        plot_ctrl_layout.addWidget(self.xaxis_box, 2, 1)
        
        plot_ctrl_layout.addWidget(QLabel("y-Axis Unit"), 3, 0)
        self.yaxis_box = QComboBox()
        self.yaxis_box.addItems(["Force", "Stress", "Elongation"])
        plot_ctrl_layout.addWidget(self.yaxis_box, 3, 1)

        plot_ctrl_box.setLayout(plot_ctrl_layout)
        tensile_controls_layout.addWidget(plot_ctrl_box)       
        
        plot_inner_layout.addWidget(container_widget)
        plot_box.setLayout(plot_inner_layout)
        top_layout_2.addWidget(plot_box)
        

    # -----------------------
    # UI builders
    # -----------------------
    def _build_control_panel_ui(self):
        # -----------------------
        # Alarms / Cruise Control
        # -----------------------
        cruise_layout = QGridLayout()
        cruise_box = QGroupBox("Alarms")
        cruise_box.setStyleSheet("font-size: 12pt;")
        cruise_inner_layout = QGridLayout()

        # Temperature alarm controls
        cruise_inner_layout.addWidget(QLabel("Min Temp [°C]:"), 0, 0)
        self.cruise_Temp_min = QLineEdit(); self.cruise_Temp_min.setText("900"); self.cruise_Temp_min.setFixedWidth(60)
        cruise_inner_layout.addWidget(self.cruise_Temp_min, 0, 1)

        cruise_inner_layout.addWidget(QLabel("Max Temp [°C]:"), 0, 2)
        self.cruise_Temp_max = QLineEdit(); self.cruise_Temp_max.setText("1000"); self.cruise_Temp_max.setFixedWidth(60)
        cruise_inner_layout.addWidget(self.cruise_Temp_max, 0, 3)

        self.set_btn_temp = QPushButton("Alarm On"); self.set_btn_temp.clicked.connect(self.toggle_Engage_temp)
        cruise_inner_layout.addWidget(self.set_btn_temp, 0, 4, 1, 2)

        # Flow alarm controls
        cruise_inner_layout.addWidget(QLabel("Flow Rate Min [%]:"), 1, 0)
        self.cruise_Flow_Min = QLineEdit(); self.cruise_Flow_Min.setText("10"); self.cruise_Flow_Min.setFixedWidth(60)
        cruise_inner_layout.addWidget(self.cruise_Flow_Min, 1, 1)

        cruise_inner_layout.addWidget(QLabel("Flow Rate Max [%]:"), 1, 2)
        self.cruise_Flow_Max = QLineEdit(); self.cruise_Flow_Max.setText("10"); self.cruise_Flow_Max.setFixedWidth(60)
        cruise_inner_layout.addWidget(self.cruise_Flow_Max, 1, 3)

        self.set_btn_flow = QPushButton("Alarm On"); self.set_btn_flow.clicked.connect(self.toggle_Engage_flow)
        cruise_inner_layout.addWidget(self.set_btn_flow, 1, 4, 1, 2)

        # Cruise Control
        self.set_btn_cc = QPushButton("Engage Cruise Control"); self.set_btn_cc.clicked.connect(self.toggle_Cruise_Control)
        cruise_inner_layout.addWidget(self.set_btn_cc, 2, 4, 1, 2)

        # Status indicators
        status_box = QGroupBox("Status"); status_box.setStyleSheet("font-size: 12pt;")
        cruise_stat_layout = QHBoxLayout()
        status_label_temp = QLabel("Temperature"); self.indicator_temp = QLabel("●"); self.indicator_temp.setStyleSheet("color: red; font-size: 18pt;")
        status_label_flow = QLabel("Flow Rates"); self.indicator_flow = QLabel("●"); self.indicator_flow.setStyleSheet("color: red; font-size: 18pt;")
        status_label_cc = QLabel("Cruise Control"); self.indicator_cc = QLabel("●"); self.indicator_cc.setStyleSheet("color: red; font-size: 18pt;")
        for w in (status_label_temp, self.indicator_temp, status_label_flow, self.indicator_flow, status_label_cc, self.indicator_cc):
            cruise_stat_layout.addWidget(w)
        status_box.setLayout(cruise_stat_layout)
        cruise_inner_layout.addWidget(status_box, 3, 0, 1, 6)
        cruise_box.setLayout(cruise_inner_layout)
        cruise_layout.addWidget(cruise_box, 3, 0)
        self.content_layout.addLayout(cruise_layout)

        # -----------------------
        # Gas channels
        # -----------------------
        gas_layout = QHBoxLayout()
        for i, gas in enumerate(gas_types):
            group_box = QGroupBox(gas); group_box.setStyleSheet("font-size: 12pt;")
            vbox = QVBoxLayout(group_box)

            # COM Port Initialization
            com_group = QGroupBox("COM Port Initialisation")
            com_layout = QGridLayout(com_group)
            com_layout.addWidget(QLabel("Unit ID"), 0, 0)
            unit_id = QLineEdit(); unit_id.setText("A"); com_layout.addWidget(unit_id, 0, 1); unit_id_widgets.append(unit_id)
            com_layout.addWidget(QLabel("Baud Rate"), 1, 0)
            baud_rate = QLineEdit(); baud_rate.setText("38400"); com_layout.addWidget(baud_rate, 1, 1); baud_rate_widgets.append(baud_rate)
            com_layout.addWidget(QLabel("COM Channel"), 2, 0)
            com_channel = QComboBox(); com_channel.addItem(""); com_layout.addWidget(com_channel, 2, 1); com_channel_widgets.append(com_channel)
            connect_btn = QPushButton("Connect"); connect_btn.clicked.connect(lambda checked, idx=i: self.connect_controller(idx)); com_layout.addWidget(connect_btn, 3, 0, 1, 2)
            disconnect_btn = QPushButton("Disconnect"); disconnect_btn.clicked.connect(lambda checked, idx=i: self.disconnect_controller(idx)); com_layout.addWidget(disconnect_btn, 4, 0, 1, 2)
            vbox.addWidget(com_group)

            # Connection Status & Flow Control
            status_group = QGroupBox("Connection Status")
            status_layout = QVBoxLayout(status_group)
            status_label = QLabel(f"{gas} connected")
            indicator = QLabel("●"); indicator.setStyleSheet("color: red; font-size: 18pt;")
            status_hbox = QHBoxLayout(); status_hbox.addWidget(status_label); status_hbox.addStretch(); status_hbox.addWidget(indicator)
            status_layout.addLayout(status_hbox)
            status_indicators.append(indicator)

            # Fuel control
            shut_group = QGroupBox("Fuel Control")
            shut_layout = QGridLayout(shut_group)
            flow_spin = QDoubleSpinBox(); flow_spin.setRange(0.0, 100.0); flow_spin.setDecimals(1); flow_spin.setSingleStep(0.5); flow_spin.setValue(0.0)
            shut_layout.addWidget(flow_spin, 0, 0); shut_layout.setColumnStretch(0, 2); flow_rate_widgets.append(flow_spin)
            unit_lbl = QLabel("SLPM"); shut_layout.addWidget(unit_lbl, 0, 1)
            apply_btn = QPushButton("Apply Flow"); apply_btn.clicked.connect(lambda checked, idx=i: self.apply_flow(idx)); shut_layout.addWidget(apply_btn, 1, 0, 1, 2)
            shut_btn = QPushButton("Shutoff"); shut_btn.clicked.connect(lambda checked, idx=i: self.shutoff_flow(idx)); shut_layout.addWidget(shut_btn, 2, 0, 1, 2)
            vbox.addWidget(status_group); vbox.addWidget(shut_group)

            # Readings
            readings_group = QGroupBox("Readings")
            readings_layout = QGridLayout(readings_group)
            labels = ["Temperature [°C]", "Setpoint [SLPM]", "Flow rate [SLPM]"]
            for j, label_text in enumerate(labels):
                lbl = QLabel(label_text); readings_layout.addWidget(lbl, j, 0)
                val_label = QLabel("0"); val_label.setFrameStyle(QLabel.Shape.Box | QLabel.Shadow.Sunken)
                readings_layout.addWidget(val_label, j, 1)
                readings_dict[gas][label_text] = val_label
            vbox.addWidget(readings_group)

            gas_layout.addWidget(group_box)
        self.content_layout.addLayout(gas_layout)

        # Emergency shutoff button
        emergency_btn = QPushButton("Emergency Shutoff")
        emergency_btn.setStyleSheet("background-color: red; color: white; font-size: 14pt;")
        emergency_btn.clicked.connect(self.emergency)
        self.content_layout.addWidget(emergency_btn)

    # -----------------------
    # Alarm / cruise logic
    # -----------------------
    def toggle_Engage_temp(self):
        if self.temp_alarm_active and not self.cruise_control_active:
            self.temp_alarm_active = False; self.temp_alarm_timer.stop()
            self.set_btn_temp.setText("Alarm On"); self.indicator_temp.setStyleSheet("color: red; font-size: 18pt;")
            self.temp_widget.min_temp_line.hide(); self.temp_widget.max_temp_line.hide()
        elif not self.temp_alarm_active and not self.cruise_control_active:
            self.temp_alarm_active = True; self.set_btn_temp.setText("Alarm Off"); self.indicator_temp.setStyleSheet("color: green; font-size: 18pt;")
            self.temp_alarm_timer.start(1000)
            try:
                self.temp_widget.update_temp_limits(self.cruise_Temp_min.text(), self.cruise_Temp_max.text())
                logger.info(f"Temperature Alarm: T Min: {self.cruise_Temp_min.text()}°C,  T Max: {self.cruise_Temp_max.text()}°C")
            except ValueError as e:
                logger.error(f"Error updating temperature limits: {e}")
        else:
            self.toggle_Cruise_Control()

    def check_temperature(self):
        if self.temp_widget.temp_val is not None:
            min_temp = self.cruise_Temp_min.text(); max_temp = self.cruise_Temp_max.text()
            if min_temp != '' and max_temp != '':
                try:
                    temp_val = self.temp_widget.temp_val
                    min_val = float(min_temp); max_val = float(max_temp)
                    self.temp_widget.update_temp_limits(min_val, max_val)
                    if temp_val < min_val or temp_val > max_val:
                        self.temp_alarm_timer.stop(); self.temp_alarm_active = False
                        logger.warning(f"Temperature {temp_val:.2f}°C out of expected range ({min_val}°C-{max_val}°C)!")
                        self._alert_window()
                        QMessageBox.warning(self, "Temperature Alarm",
                                            f"Temperature {temp_val:.2f}°C is out of range!\nResetting Alarm")
                        self.set_btn_temp.setText("Alarm On"); self.indicator_temp.setStyleSheet("color: red; font-size: 18pt;")
                        self.temp_widget.min_temp_line.hide(); self.temp_widget.max_temp_line.hide()
                except ValueError as e:
                    logger.error(f"Temperature check error: {e}")
                    self.temp_alarm_timer.stop()

    def toggle_Engage_flow(self):
        if self.flow_alarm_active and not self.cruise_control_active:
            self.flow_alarm_active = False; self.flow_alarm_timer.stop()
            self.set_btn_flow.setText("Alarm On"); self.indicator_flow.setStyleSheet("color: red; font-size: 18pt;")
        elif not self.flow_alarm_active and not self.cruise_control_active:
            self.flow_alarm_active = True; self.set_btn_flow.setText("Alarm Off"); self.indicator_flow.setStyleSheet("color: green; font-size: 18pt;")
            self.flow_alarm_timer.start(1000)
            logger.info(f"Flow Alarm: Flow Rate Min: {100-float(self.cruise_Flow_Min.text()):.0f}%, Flow Rate Max: {100+float(self.cruise_Flow_Max.text()):.0f}%")
        else:
            self.toggle_Cruise_Control()

    def check_flows(self):
        try:
            for index in range(len(gas_types)):
                if not connect_status[index]:
                    continue
                flow_rate_val = float(readings_dict[gas_types[index]]["Flow rate [SLPM]"].text())
                setpoint_val = float(readings_dict[gas_types[index]]["Setpoint [SLPM]"].text())
                if setpoint_val != 0:
                    low = setpoint_val*(1 - float(self.cruise_Flow_Min.text())/100)
                    high = setpoint_val*(1 + float(self.cruise_Flow_Max.text())/100)
                    if flow_rate_val < low or flow_rate_val > high:
                        self.flow_alarm_timer.stop(); self.flow_alarm_active = False
                        logger.warning(f"{gas_types[index]} flow out of range! (flow: {flow_rate_val:.2f}, set: {setpoint_val:.2f}, limits: {low:.2f}-{high:.2f})")
                        self._alert_window()
                        QMessageBox.warning(self, "Flow Rate Alarm", f"{gas_types[index]} flow is out of range!")
                        self.set_btn_flow.setText("Alarm On"); self.indicator_flow.setStyleSheet("color: red; font-size: 18pt;")
                        break
        except ValueError as e:
            logger.error(f"Flow alarm error: {e}")
            self.flow_alarm_timer.stop()

    def toggle_Cruise_Control(self):
        if self.cruise_control_active:
            self.cruise_control_active = False; self.cruise_control_timer.stop()
            self.set_btn_cc.setText("Engage Cruise Control"); self.indicator_cc.setStyleSheet("color: red; font-size: 18pt;")
            self.set_btn_flow.setText("Alarm On"); self.indicator_flow.setStyleSheet("color: red; font-size: 18pt;")
            self.set_btn_temp.setText("Alarm On"); self.indicator_temp.setStyleSheet("color: red; font-size: 18pt;")
            self.temp_widget.min_temp_line.hide(); self.temp_widget.max_temp_line.hide()
            logger.info("Cruise Control disengaged")
        else:
            reply = QMessageBox.question(self, "Confirmation",
                                         "Cruise Control will automatically monitor and shut off fuels if flows and/or temperatures are outside of range. Continue?",
                                         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Yes:
                self.cruise_control_active = True; self.set_btn_cc.setText("Disengage Cruise Control")
                if self.flow_alarm_active:
                    self.flow_alarm_active = False; self.flow_alarm_timer.stop()
                if self.temp_alarm_active:
                    self.temp_alarm_active = False; self.temp_alarm_timer.stop()
                self.indicator_cc.setStyleSheet("color: green; font-size: 18pt;")
                self.cruise_control_timer.start(1000)
                self.set_btn_flow.setText("Alarm Off"); self.indicator_flow.setStyleSheet("color: green; font-size: 18pt;")
                self.set_btn_temp.setText("Alarm Off"); self.indicator_temp.setStyleSheet("color: green; font-size: 18pt;")
                try:
                    self.temp_widget.update_temp_limits(self.cruise_Temp_min.text(), self.cruise_Temp_max.text())
                except ValueError as e:
                    logger.error(f"Error updating temperature limits: {e}")
                logger.info("Cruise Control engaged")
                logger.info(f"Cruise Control info: T Min: {self.cruise_Temp_min.text()}°C,  T Max: {self.cruise_Temp_max.text()}°C")
                logger.info(f"Cruise Control info: Flow Rate Min: {100-float(self.cruise_Flow_Min.text()):.0f}%, Flow Rate Max: {100+float(self.cruise_Flow_Max.text()):.0f}%")

    def control_cruise(self):
        try:
            for index in range(len(gas_types)):
                if not connect_status[index]:
                    continue
                flow_rate_val = float(readings_dict[gas_types[index]]["Flow rate [SLPM]"].text())
                setpoint_val = float(readings_dict[gas_types[index]]["Setpoint [SLPM]"].text())
                if setpoint_val != 0:
                    low = setpoint_val*(1 - float(self.cruise_Flow_Min.text())/100)
                    high = setpoint_val*(1 + float(self.cruise_Flow_Max.text())/100)
                    if flow_rate_val < low or flow_rate_val > high:
                        self.cruise_control_timer.stop(); self.cruise_control_active = False
                        self.emergency()
                        logger.warning(f"{gas_types[index]} flow out of expected range (flow: {flow_rate_val:.2f}, set: {setpoint_val:.2f}, limits: {low:.2f}-{high:.2f}). Shutting Down!")
                        self._alert_window()
                        QMessageBox.warning(self, "Flow Rate Alarm", f"{gas_types[index]} flow is out of expected range. Shutting Down!")
                        self.set_btn_cc.setText("Engage Cruise Control"); self.indicator_cc.setStyleSheet("color: red; font-size: 18pt;")
                        self.set_btn_flow.setText("Alarm On"); self.indicator_flow.setStyleSheet("color: red; font-size: 18pt;")
                        self.set_btn_temp.setText("Alarm On"); self.indicator_temp.setStyleSheet("color: red; font-size: 18pt;")
                        self.temp_widget.min_temp_line.hide(); self.temp_widget.max_temp_line.hide()
                        break
        except ValueError as e:
            logger.error(f"Flow alarm error: {e}")
            self.cruise_control_timer.stop(); self.cruise_control_active = False

        if self.temp_widget.temp_val is not None:
            min_temp = self.cruise_Temp_min.text(); max_temp = self.cruise_Temp_max.text()
            if min_temp != '' and max_temp != '':
                try:
                    temp_val = self.temp_widget.temp_val
                    min_val = float(min_temp); max_val = float(max_temp)
                    if temp_val < min_val or temp_val > max_val:
                        self.cruise_control_timer.stop(); self.cruise_control_active = False
                        self.emergency()
                        logger.warning(f"Temperature {temp_val:.2f}°C out of expected range ({min_val}°C - {max_val}°C). Shutting Down!")
                        self._alert_window()
                        QMessageBox.warning(self, "Temperature Alarm", f"Temperature {temp_val:.2f}°C is out of range. Shutting Down!")
                        self.set_btn_cc.setText("Engage Cruise Control"); self.indicator_cc.setStyleSheet("color: red; font-size: 18pt;")
                        self.set_btn_flow.setText("Alarm On"); self.indicator_flow.setStyleSheet("color: red; font-size: 18pt;")
                        self.set_btn_temp.setText("Alarm On"); self.indicator_temp.setStyleSheet("color: red; font-size: 18pt;")
                        self.temp_widget.min_temp_line.hide(); self.temp_widget.max_temp_line.hide()
                except ValueError as e:
                    logger.error(f"Temperature check error: {e}")
                    self.cruise_control_timer.stop(); self.cruise_control_active = False

    def format_time(self, seconds):
        if seconds < 60:
            return f"{seconds} s"
        elif seconds < 3600:
            return f"{seconds/60} min"
        else:
            return f"{seconds/3600} h"


    def start_tensile_test(self):
        
        if len(self.tensile_time_data)  > 0:
            reply = QMessageBox.question(self, "Confirmation",
                            "Overwrite tensile data and start new test?",
                            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply != QMessageBox.StandardButton.Yes:
                return  # user cancelled
        
        if self.gear_ratio == "None":
            self.ratio = 1.0
        else:
            self.ratio = 50

        try:
            if float(self.MMotor_speed.text()) >0:            
                if self.ratio*float(self.MMotor_speed.text()) >47.5: #190 rpm
                    QMessageBox.critical(self, "Error", "Maximum allowed speed: 0.95 mm/min (50:1 gearbox),\n47.5 mm/min (no gearbox).")
                    logger.error('Selected motor speed is too high!')  
                    return
                else:
                    self.motor_speed = float(self.MMotor_speed.text()) #mm/min
            else:
                QMessageBox.critical(self, "Error", "Motor speed must be greater than 0 mm/min.")
                logger.error('Motor speed must be greater than 0 mm/min')  
                return
        except ValueError as e:
            logger.error(f"Invalid motor speed: {e}")
            return 
        rpm = self.motor_speed / self.screwjack_speed * self.ratio
        self.tensile_current_rpm = rpm #store rpm for pause button
        
        selected = self.tensile_combobox.currentText()
        if not selected:
            QMessageBox.warning(self, "Warning", "Please select a COM port.")
            return

        com_port = selected.split(" - ")[0]
        try:
            # find already-open connection (prefer self.temp_ser)
            conn = None
            if hasattr(self, "temp_ser") and self.temp_ser and getattr(self.temp_ser, "is_open", False) and getattr(self.temp_ser, "port", None) == com_port:
                conn = self.temp_ser
            else:
                for sc in serial_connections:
                    if sc and getattr(sc, "port", None) == com_port and sc.is_open:
                        conn = sc
                        break

            if conn is None:
                QMessageBox.critical(self, "Error", "No open serial connection for selected COM port.")
                return

            # prepare commands - device-specific; keep newline/terminator as device expects
            rpm_cmd = f"RPM {rpm}\n".encode('ascii')
            cmds = [f"MICRO {self.micro}\n", f"DIR {self.tensile_dir}\n", b"EN 1\n", rpm_cmd, f"CAL {self.calibration_factor}\n"]
            self.speed_label.setText(f"{self.motor_speed}")
            
            self.start_speed = self.motor_speed

            # if an existing worker is running, stop it first
            if getattr(self, "tensile_worker", None):
                try:
                    self.stop_tensile_test()
                except Exception:
                    pass

            # create worker + thread, run in background until stop pressed
            self.tensile_thread = QThread()
            self.tensile_worker = TensileWorker(conn, cmds)
            self.tensile_worker.moveToThread(self.tensile_thread)
            self.tensile_thread.started.connect(self.tensile_worker.run)
            self.tensile_worker.finished.connect(self.tensile_thread.quit)
            self.tensile_worker.error.connect(lambda e: logger.error(f"Tensile worker error: {e}"))
            self.tensile_thread.finished.connect(lambda: setattr(self, "tensile_worker", None))
            self.tensile_thread.start()
            self.tensile_timer.start(self.tensile_update_interval)
            logger.info(f"Tensile test started with {self.motor_speed} mm/min.")
            self.btn_start_tens.setEnabled(False) 
            self.btn_stop_tens.setEnabled(True)
            self.btn_pause_tens.setEnabled(True)
            self.btn_apply_speed.setEnabled(True)
            self.stage_setup_btn.setEnabled(False)

            try:
                self.clear_lists()
                self.tensile_start_time = time.time()
            except Exception as e:
                logger.error(f"Failed to start tensile plot timer: {e}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Couldn't start tensile test: {e}")

        # Create force reader thread (separate from command thread)
        self.force_reader_thread = QThread()
        self.force_reader = TensileForceReader(conn, self.tensile_update_interval)
        self.force_reader.moveToThread(self.force_reader_thread)
        self.force_reader_thread.started.connect(self.force_reader.run)
        self.force_reader.force_updated.connect(self.handle_force_data)
        self.force_reader.error.connect(lambda e: logger.error(f"Force reader: {e}"))
        self.force_reader_thread.start()
    

    def handle_force_data(self, force_val):
        self.force_val_raw = force_val
        self.force_val = self.force_val_raw - self.zero_F
        self.force_label.setText(f"{self.force_val_raw:.1f}")

    def pause_tensile_test(self):
        """
        Toggle pause/resume:
        - Pause: send RPM 0 to controller (motor stops), keep timers/plot running.
                 save current rpm/motor_speed so resume can restore.
        - Resume: restore saved_motor_speed (if user applied a new speed while paused),
                  otherwise restore the rpm that was active before pause.
        """
        # require an active test
        if not getattr(self, "tensile_worker", None):
            QMessageBox.information(self, "Info", "No tensile test is running.")
            return

        conn = getattr(self.tensile_worker, "conn", None)
        if conn is None:
            QMessageBox.warning(self, "Warning", "No serial connection available to send pause/resume commands.")
            return

        # two flags exist in code historically; keep both in sync
        paused = getattr(self, "tensile_paused", False) or getattr(self, "paused", False)

        if not paused:
            # Pause: save rpm & motor_speed, send RPM 0
            self._tensile_saved_rpm = getattr(self, "tensile_current_rpm", None)
            # also keep a copy of the user-facing motor_speed if present
            self._tensile_saved_motor_speed = getattr(self, "motor_speed", None)
            with serial_lock:
                try:
                    conn.write(b"RPM 0\n")
                except Exception as e:
                    logger.error(f"Failed to send RPM 0: {e}")
            # update UI to show paused speed
            try:
                self.speed_label.setText("0")
            except Exception:
                pass
            # set flags and button text
            self.tensile_paused = True
            try:
                self.btn_pause_tens.setText("Resume")
            except Exception:
                pass
            logger.info("Tensile test paused")
        else:
            # Resume: prefer saved_motor_speed (user applied while paused), else restore previous rpm
            rpm_to_restore = None
            # if user applied a new speed while paused, self.saved_motor_speed (mm/min) will be set
            if getattr(self, "saved_motor_speed", None) is not None:
                try:
                    
                    ratio_val = 50.0 if ("50" in self.gear_ratio) else 1.0
                         
                    # compute rpm from saved motor speed
                    rpm_to_restore = float(self.saved_motor_speed) / self.screwjack_speed * float(ratio_val)
                    restore_speed_label = f"{self.saved_motor_speed}"
                except Exception:
                    rpm_to_restore = getattr(self, "_tensile_saved_rpm", None)
                    restore_speed_label = f"{getattr(self, '_tensile_saved_motor_speed', getattr(self, 'motor_speed', 0))}"
            else:
                rpm_to_restore = getattr(self, "_tensile_saved_rpm", None)
                restore_speed_label = f"{getattr(self, '_tensile_saved_motor_speed', getattr(self, 'motor_speed', 0))}"

            if rpm_to_restore is None:
                QMessageBox.warning(self, "Warning", "No saved RPM available to resume.")
                return

            # send restore command
            with serial_lock:
                try:
                    conn.write(f"RPM {rpm_to_restore}\n".encode("ascii"))
                except Exception as e:
                    logger.error(f"Failed to restore RPM {rpm_to_restore}: {e}")

            # update UI label
            try:
                self.speed_label.setText(restore_speed_label)
            except Exception:
                pass

            # clear saved pending speed (if any)
            if getattr(self, "saved_motor_speed", None) is not None:
                self.saved_motor_speed = None

            # clear internal saved rpm copy
            try:
                delattr = False
                if hasattr(self, "_tensile_saved_rpm"):
                    del self._tensile_saved_rpm
                if hasattr(self, "_tensile_saved_motor_speed"):
                    del self._tensile_saved_motor_speed
            except Exception:
                pass

            self.tensile_paused = False
            try:
                self.btn_pause_tens.setText("Pause")
            except Exception:
                pass
            logger.info("Tensile test resumed")

    def stop_tensile_test(self):
        """Stop the running tensile worker and send disable commands to the controller."""
        try:
            # If no worker exists, notify and return
            if not getattr(self, "tensile_worker", None):
                logger.info("No tensile test was running.")
                return

            # Confirm stop with user
            reply = QMessageBox.question(self, "Confirmation",
                                         "End Tensile Test?",
                                         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply != QMessageBox.StandardButton.Yes:
                return  # user cancelled
            else: 
                self.abort_test()
        except Exception as e:
            logger.error(f"Error stopping tensile test: {e}")

    def abort_test(self):
        # Stop force reader thread
        try:
            if hasattr(self, "force_reader"):
                self.force_reader.stop()
                self.force_reader_thread.quit()
                self.force_reader_thread.wait(2000)
        except Exception:
            pass

        # stop simulated plotting timer
        try:
            if hasattr(self, "tensile_timer"):
                self.tensile_timer.stop()
        except Exception:
            pass

        # request worker stop
        try:
            self.tensile_worker.stop()
        except Exception:
            pass

        # send immediate stop/disable commands under lock (best-effort)
        with serial_lock:
            try:
                conn = getattr(self.tensile_worker, "conn", None)
                if conn:
                    conn.write(b"RPM 0\n")
                    conn.write(b"EN 0\n")
                    self.speed_label.setText("0")
            except Exception:
                pass

        # wait for thread to finish
        try:
            if getattr(self, "tensile_thread", None):
                self.tensile_thread.quit()
                self.tensile_thread.wait(2000)
        except Exception:
            pass

        logger.info("Tensile test stopped.")
        self.tensile_timer.stop()
        self.btn_start_tens.setEnabled(True) 
        self.btn_stop_tens.setEnabled(False)
        self.btn_pause_tens.setEnabled(False)
        self.btn_apply_speed.setEnabled(False)
        self.stage_setup_btn.setEnabled(True)
        self.tensile_paused = False
        self.btn_pause_tens.setText("Pause")


    def update_tensile_data(self):
        """Update plot with pre-fetched force data (no serial I/O here)"""
        if not getattr(self, "tensile_worker", None):
            return
        
        # All serial reads are now done in background thread -> just update plot
        self.elon = float(self.speed_label.text()) * self.tensile_update_interval*1e-3/60 - self.zero_elon
        self.elongation_array_ind.append(self.elon)
        self.elongation_array.append(np.sum(self.elongation_array_ind))
        
        elapsed = time.time() - self.tensile_start_time
        self.time_lbl.setText(f"{np.round(float(self.format_time(elapsed).split(' ')[0]),1)}")
        self.time_unit_lbl.setText(f"{self.format_time(elapsed).split(' ')[1]}")
        self.tensile_time_data.append(elapsed)
        self.tensile_force_values.append(self.force_val)
    
        if abs(self.force_val_raw) >self.force_lim: #aborts test if limit is exceeded
            logger.warning(f'Force of {abs(np.round(self.force_val_raw,1))} N exceeded limit of {np.round(self.force_lim,0)} N. Aborting.')
            self.abort_test()
            self._alert_window()
            QMessageBox.critical(self, "Error", f'Force of {abs(np.round(self.force_val_raw,1))} N exceeded limit of {np.round(self.force_lim,0)} N. Aborting.')
        self.speed_array.append(self.speed_label.text())
        self.sample_temp_array.append(float(self.sample_temp_label.text()))
        self.loadcell_temp_array.append(float(self.loadcell_temp_label.text()))
        self.ext_label.setText(f"{np.round(self.elongation_array[-1],4)}")
        self.tensile_stress_values.append(float(self.force_val)/((float(self.sample_gauge.text())*float(self.sample_thickness.text()))))

    
    def update_tensile_plot_axes(self):
        if not self.tensile_time_data:
            return
        max_t = max(self.tensile_time_data)  # Reformat time
        if max_t < 60:
            factor = 1.0
            self.xlabel = "Time (s)"
        elif max_t < 3600:
            factor = 1.0 / 60.0
            self.xlabel = "Time (min)"
        else:
            factor = 1.0 / 3600.0
            self.xlabel = "Time (h)"

        # scale time data for plotting and update axis label
        scaled_time = [t * factor for t in self.tensile_time_data]

        # update plot
        if self.xaxis_box.currentText() == "Time":
            self.xarr =  scaled_time
        else :
            self.xarr =  self.elongation_array
            self.xlabel = "Elongation (mm)"

        if self.yaxis_box.currentText() == "Force":
            self.yarr =  self.tensile_force_values
            self.ylabel = "Force (N)"
            
        elif self.yaxis_box.currentText() == "Stress":
            self.yarr =  self.tensile_stress_values
            self.ylabel = "Engineering Stress (MPa)"
        else :
            self.yarr =  self.elongation_array
            self.ylabel = "Elongation (mm)"
            
        try:
            self.plot_tens_curve.setData(self.xarr, self.yarr)
        except Exception as e:
            logger.debug(f"Tensile plot update failed: {e}")

        try:
            self.plot_widget.setLabel("bottom", self.xlabel, **{"color": "black", "font-size": "18px"})
            self.plot_widget.setLabel("left", self.ylabel, **{"color": "black", "font-size": "18px"})
        except Exception:
            pass           

    def connect_controller(self, index):
        self.connect_thread = SerialConnectThread(index)
        self.connect_thread.finished.connect(lambda result: self.handle_connect_result(index, result))
        self.connect_thread.start()
        # Timeout guard
        self.connect_timer = QTimer(self); self.connect_timer.setSingleShot(True)
        self.connect_timer.timeout.connect(self.handle_connect_timeout)
        self.connect_timer.start(3000)

    def connect_tensile_arduino(self):
        selected = self.tensile_combobox.currentText()
        if not selected:
            QMessageBox.warning(self, "Warning", "Please select a COM port.")
            return

        com_port = selected.split(" - ")[0]
        
        # Create worker and thread
        self.tensile_thread = QThread()
        self.tensile_worker = DeviceSerialWorker(com_port, self.tensile_baud_rate)
        
        # Setup connections
        self.tensile_worker.moveToThread(self.tensile_thread)
        self.tensile_thread.started.connect(self.tensile_worker.run)
        self.tensile_worker.data_received.connect(self.handle_tensile_data)
        self.tensile_worker.error.connect(lambda e: logger.error(f"Tensile error: {e}"))
        
        # Start thread
        self.tensile_thread.start()
        logger.info("Tensile controller connected")

    def disconnect_tensile_arduino(self):
        if hasattr(self, "tensile_worker"):
            self.tensile_worker.stop()
            self.tensile_thread.quit()
            self.tensile_thread.wait()
            logger.info("Tensile controller disconnected")
        
    def handle_tensile_data(self, data: str):
        """Handle data received from tensile arduino"""
        try:
            m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", data)
            if m:
                self.force_val_raw = float(m.group(0))
                self.force_val = self.force_val_raw - self.zero_F
                self.force_label.setText(f"{self.force_val_raw:.1f}")
        except Exception as e:
            logger.error(f"Error parsing tensile data: {e}")

    def handle_connect_timeout(self):
        QMessageBox.warning(self, "Timeout", "Connection attempt timed out.")

    def handle_connect_result(self, index, result):
        try:
            self.connect_timer.stop()
        except Exception:
            pass
        if "error" in result:
            QMessageBox.critical(self, "Connection Failed", result["error"]); return
        com_port = result["com_port"]; serial_conn = result["serial_conn"]; stored_gas = result["stored_gas"]
        if stored_gas == gas_types[index]:
            status_indicators[index].setStyleSheet("color: green; font-size: 18pt;")
            serial_connections[index] = serial_conn; connect_status[index] = True
            logger.info(f"Connected to {gas_types[index]} controller on {com_port}.")
        else:
            reply = QMessageBox.question(
                self, "Confirm Connection",
                f"Controller calibrated for {stored_gas} but selected for {gas_types[index]}. Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                status_indicators[index].setStyleSheet("color: green; font-size: 18pt;")
                serial_connections[index] = serial_conn; connect_status[index] = True
                logger.info(f"Connected to {gas_types[index]} controller on {com_port}.")
            else:
                logger.info("Aborted connection"); serial_conn.close(); self.disconnect_controller(index)

    def disconnect_controller(self, index):
        if serial_connections[index] and serial_connections[index].is_open:
            with serial_lock:
                serial_connections[index].close(); serial_connections[index] = None
        status_indicators[index].setStyleSheet("color: red; font-size: 18pt;")
        connect_status[index] = False
        logger.info(f"{gas_types[index]} controller disconnected")

    def apply_flow(self, index):
        try:
            setpoint = float(flow_rate_widgets[index].value())
            unit_id = unit_id_widgets[index].text().strip()
            command = f"{unit_id}S {setpoint:.2f}\r"
            max_attempts = 3; success = False
            for attempt in range(max_attempts):
                response = send_ascii_command(serial_connections[index], command)
                if response:
                    logger.info(f"{gas_types[index]} flow set to {setpoint} SLPM."); success = True; break
                else:
                    logger.warning(f"Attempt {attempt+1} to set {gas_types[index]} flow failed.")
            if not success:
                logger.error(f"Failed to apply {gas_types[index]} flow after {max_attempts} attempts.")
        except ValueError:
            logger.error(f"{gas_types[index]} flow must be numeric.")

    def emergency(self):
        logger.info("Emergency button pressed")
        for i in range(len(flow_rate_widgets)):
            flow_rate_widgets[i].setValue(0)
            self.apply_flow(i)

    def shutoff_flow(self, index):
        reply = QMessageBox.question(self, "Confirm Shutoff", f"Confirm {gas_types[index]} shutoff?",
                                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            logger.info(f"{gas_types[index]} Shutoff pressed")
            flow_rate_widgets[index].setValue(0)
            self.apply_flow(index)

    def update_ui(self):
        try:
            while not data_queue.empty():
                try:
                    index, controller_data = data_queue.get_nowait()
                    for label, value in controller_data.items():
                        if label in readings_dict[gas_types[index]]:
                            readings_dict[gas_types[index]][label].setText(value)
                    try:
                        flow_rate_val = float(readings_dict[gas_types[index]]["Flow rate [SLPM]"].text())
                        setpoint_val = float(readings_dict[gas_types[index]]["Setpoint [SLPM]"].text())
                        if flow_rate_val <= 0.9 * setpoint_val:
                            readings_dict[gas_types[index]]["Flow rate [SLPM]"].setStyleSheet("color: red;")
                            if abs(flow_rate_val) >= 0.05:
                                now = time.time()
                                if now - self._flow_warn_times.get(index, 0) >= 3.0:
                                    self._flow_warn_times[index] = now
                                    logger.warning(f"{gas_types[index]} flow below threshold!")
                        else:
                            readings_dict[gas_types[index]]["Flow rate [SLPM]"].setStyleSheet("color: white;")
                    except ValueError:
                        pass
                except queue.Empty:
                    break
            self.update_tensile_plot_axes() #updates tensile plot so that graph can be plotted even when no tensile test is running
        except Exception as e:
            logger.error(f"Update UI error: {e}")

    def update_com_channels(self):
        try:
            ports = list_ports.comports()
            com_ports = sorted([f"{port.device} - {port.description}" for port in ports])
            for widget in com_channel_widgets:
                if not widget.view().isVisible():
                    current = widget.currentText()
                    widget.blockSignals(True)
                    widget.clear()
                    widget.addItems(com_ports)
                    if current in com_ports:
                        widget.setCurrentText(current)
                    widget.blockSignals(False)

            if not self.tensile_combobox.view().isVisible():
                current = self.tensile_combobox.currentText()
                self.tensile_combobox.blockSignals(True)
                self.tensile_combobox.clear()
                self.tensile_combobox.addItems(com_ports)
                if current in com_ports:
                    self.tensile_combobox.setCurrentText(current)
                self.tensile_combobox.blockSignals(False)

        except Exception as e:
            logger.error(f"COM channel update error: {e}")

    def update_tensile_readings(self):
        if not getattr(self, "tensile_worker", None):
            self.tensile_timer.stop()
            return

    def update_sample_temp_label(self, temp):
            self.sample_temp_label.setText(f"{temp:.2f}")  # Update the sample temperature label

    def apply__motor_speed(self):
        
        if not getattr(self, "tensile_worker", None):
            QMessageBox.information(self, "Info", "No tensile test running.")
            return
        
        conn = getattr(self.tensile_worker, "conn", None)
        if not conn is None:
            
            self.ratio = 1.0 if self.gear_ratio == "None" else 50
            try:
                if float(self.MMotor_speed.text()) >0:            
                    if self.ratio*float(self.MMotor_speed.text()) >47.5: #190 rpm
                        QMessageBox.critical(self, "Error", "Maximum allowed speed: 0.95 mm/min (50:1 gearbox),\n47.5 mm/min (no gearbox).")
                        logger.error('Selected motor speed is too high!')  
                        return
                    else:
                        self.motor_speed = float(self.MMotor_speed.text()) #mm/min
                else:
                    QMessageBox.critical(self, "Error", "Motor speed must be greater than 0 mm/min.")
                    logger.error('Motor speed must be greater than 0 mm/min')                
                    return
            except ValueError as e:
                logger.error(f"Invalid motor speed: {e}")
                return 
            rpm = self.motor_speed / self.screwjack_speed * self.ratio
            self.tensile_current_rpm = rpm 
            
            if self.tensile_paused:
                self.saved_motor_speed = self.motor_speed  # Save the new speed if paused
                logger.info(f"Motor speed saved: {self.motor_speed} mm/min (will apply on resume).")
            else:
                # Send the new speed to the motor immediately
                conn.write(f"DIR {self.tensile_dir}\n".encode('ascii'))
                conn.write(f"RPM {rpm}\n".encode('ascii'))  # Send RPM command to the motor
                self.speed_label.setText(f"{self.motor_speed}")
                logger.info(f"Motor speed set to {self.motor_speed} mm/min")

    def select_csv_file(self):
        csv_file_path, _ = QFileDialog.getSaveFileName(self, "Select Log File", "", "CSV files (*.csv)")
        if csv_file_path:
            # store path only (touch file so it exists). do not replace file handle with writer
            self.tensile_csv_path = csv_file_path
            try:
                open(csv_file_path, "a", newline="").close()
                logger.info(f"CSV file created: {csv_file_path}")
            except Exception as e:
                logger.error(f"Failed to create CSV file: {e}")
                QMessageBox.critical(self, "CSV", f"Failed to create CSV file: {e}")


    def write_tensile_data(self):
        if len(self.tensile_time_data) > 0:
            path = getattr(self, "tensile_csv_path", None)
            if path is None:
                QMessageBox.warning(self, "CSV", "No CSV file selected.")
                return
            try:
                with open(path, "a", newline="") as fh:
                    fh.truncate(0)
                    writer = csv.writer(fh)

                    writer.writerow(["Date", self.tensile_start_date])
                    writer.writerow(["Time", datetime.now().strftime("%H:%M:%S")])
                    writer.writerow(["Gearbox", self.gear_ratio])
                    writer.writerow(["Loadcell", self.loadcell])
                    writer.writerow(["calibration factor", self.calibration_factor])
                    writer.writerow(["sampling interval", f"{self.tensile_update_interval} ms"])
                    writer.writerow(["Start speed", f"{self.start_speed} mm/min"])
                    writer.writerow(["Test length", f"{np.round(max(self.tensile_time_data),2)} s"])
                    writer.writerow(["Sample Thickness", f"{self.sample_thickness.text()} mm"])
                    writer.writerow(["Gauge Width", f"{self.sample_gauge.text()} mm"])
                    writer.writerow("")
                    
                    tensile_data_all = zip(np.round(self.tensile_time_data,1),
                                           np.round(self.tensile_force_values,2),
                                           np.round(self.tensile_stress_values,2),
                                           np.round(self.elongation_array,4),
                                           self.speed_array,
                                           np.round(self.sample_temp_array,2),
                                           np.round(self.loadcell_temp_array,2))
                    
                    writer.writerow(["Time [s]", "Force [N]", "Eng Stress [MPa]", "Elongation [mm]", "Motor speed [mm/min]", "Sample temperature [°C]", "Load cell temperature [°C]"])
                    for row in tensile_data_all:
                        writer.writerow(row)
                    
                logger.info(f"Wrote tensile metadata to {path}")
            except Exception as e:
                logger.error(f"Failed to write tensile data: {e}")
                QMessageBox.critical(self, "CSV", f"Failed to write tensile data: {e}")
        else:
            QMessageBox.warning(self, "Data", "No tensile data available.\nStart a run first.")

    def clear_lists(self):
        self.tensile_time_data.clear()
        self.tensile_force_values.clear()
        self.tensile_stress_values.clear()
        self.elongation_array.clear()
        self.elongation_array_ind.clear()
        self.speed_array.clear()
        self.sample_temp_array.clear()
        self.loadcell_temp_array.clear()

    def open_jaws(self):

        # don't allow opening jaws while a run is active
        if getattr(self, "tensile_worker", None):
            QMessageBox.information(self, "Info", "Please stop tensile test before opening jaws.")
            return

        selected = self.tensile_combobox.currentText()
        if not selected:
            QMessageBox.warning(self, "Warning", "Please select a COM port.")
            return
        com_port = selected.split(" - ")[0]

        # find an open serial connection for the selected COM port
        conn = None
        if hasattr(self, "temp_ser") and self.temp_ser and getattr(self.temp_ser, "is_open", False) and getattr(self.temp_ser, "port", None) == com_port:
            conn = self.temp_ser
        else:
            for sc in serial_connections:
                if sc and getattr(sc, "port", None) == com_port and sc.is_open:
                    conn = sc
                    break

        if conn is None:
            QMessageBox.critical(self, "Error", "No open serial connection for selected COM port.")
            return
        
        if float(self.speed_label.text()) != 0 and self.close_jaw_var == 1:
            QMessageBox.critical(self, "Error", "Please stop motor movement before opening jaws.")
            return

        if self.jaw_ctrl_var == 0:
            
            self.ratio = 1.0 if self.gear_ratio == "None" else 50
            try:
                if float(self.MMotor_speed.text()) >0:            
                    if self.ratio*float(self.MMotor_speed.text()) >47.5: #190 rpm
                        QMessageBox.critical(self, "Error", "Maximum allowed speed: 0.95 mm/min (50:1 gearbox),\n47.5 mm/min (no gearbox).")
                        logger.error('Selected motor speed is too high!')  
                        return
                    else:
                        self.motor_speed = float(self.MMotor_speed.text()) #mm/min
                else:
                    QMessageBox.critical(self, "Error", "Motor speed must be greater than 0 mm/min.")
                    logger.error('Motor speed must be greater than 0 mm/min!')  
                    return
            except ValueError as e:
                logger.error(f"Invalid motor speed: {e}")
                return 
            rpm = self.motor_speed / self.screwjack_speed * self.ratio
            self.tensile_current_rpm = rpm 
            
            # Send the new speed to the motor immediately
            conn.write(f"EN 1\n".encode('ascii'))
            conn.write(f"DIR {1 if self.tensile_dir ==0 else 0}\n".encode('ascii')) #other direction to tensile
            conn.write(f"RPM {rpm}\n".encode('ascii'))  # Send RPM command to the motor
            self.speed_label.setText(f"{self.motor_speed}")
            logger.info(f"Jaws opening")

            self.jaw_ctrl_var =1
            self.open_jaw_var = 1

        else:
            conn.write(f"DIR {1 if self.tensile_dir ==0 else 0}\n".encode('ascii')) #other direction to tensile
            conn.write(f"RPM 0\n".encode('ascii'))  # Send RPM command to the motor
            conn.write(f"EN 0\n".encode('ascii'))
            self.speed_label.setText("0 mm/min")
            logger.info(f"Jaw movement stopping")

            self.jaw_ctrl_var = 0
            self.open_jaw_var = 0


    def close_jaws(self):
        # don't allow opening jaws while a run is active
        if getattr(self, "tensile_worker", None):
            QMessageBox.information(self, "Info", "Please stop tensile test before closing jaws.")
            return

        selected = self.tensile_combobox.currentText()
        if not selected:
            QMessageBox.warning(self, "Warning", "Please select a COM port.")
            return
        com_port = selected.split(" - ")[0]

        # find an open serial connection for the selected COM port
        conn = None
        if hasattr(self, "temp_ser") and self.temp_ser and getattr(self.temp_ser, "is_open", False) and getattr(self.temp_ser, "port", None) == com_port:
            conn = self.temp_ser
        else:
            for sc in serial_connections:
                if sc and getattr(sc, "port", None) == com_port and sc.is_open:
                    conn = sc
                    break

        if conn is None:
            QMessageBox.critical(self, "Error", "No open serial connection for selected COM port.")
            return
        
        if float(self.speed_label.text()) != 0 and self.open_jaw_var == 1:
            QMessageBox.critical(self, "Error", "Please stop motor movement before closing jaws.")
            return

        if self.jaw_ctrl_var == 0:
            
            self.ratio = 1.0 if self.gear_ratio == "None" else 50
            try:
                if float(self.MMotor_speed.text()) >0:            
                    if self.ratio*float(self.MMotor_speed.text()) >47.5: #190 rpm
                        QMessageBox.critical(self, "Error", "Maximum allowed speed: 0.95 mm/min (50:1 gearbox),\n47.5 mm/min (no gearbox).")
                        logger.error('Selected motor speed is too high!')  
                        return
                    else:
                        self.motor_speed = float(self.MMotor_speed.text()) #mm/min
                else:
                    QMessageBox.critical(self, "Error", "Motor speed must be greater than 0 mm/min!")
                    logger.error('Motor speed must be greater than 0 mm/min.')  
                    return
            except ValueError as e:
                logger.error(f"Invalid motor speed: {e}")
                return 
            rpm = self.motor_speed / self.screwjack_speed * self.ratio
            self.tensile_current_rpm = rpm 
            
            # Send the new speed to the motor immediately
            conn.write(f"EN 1\n".encode('ascii'))
            conn.write(f"DIR {self.tensile_dir}\n".encode('ascii')) #tensile direction
            conn.write(f"RPM {rpm}\n".encode('ascii'))  # Send RPM command to the motor
            self.speed_label.setText(f"{self.motor_speed}")
            logger.info(f"Jaws opening")

            self.close_jaw_var = 1
            self.jaw_ctrl_var =1

        else:
            conn.write(f"DIR {self.tensile_dir}\n".encode('ascii')) #tensile direction
            conn.write(f"RPM 0\n".encode('ascii'))  # Send RPM command to the motor
            conn.write(f"EN 0\n".encode('ascii'))
            self.speed_label.setText("0")
            logger.info(f"Jaw movement stopping")

            self.jaw_ctrl_var = 0
            self.close_jaw_var = 0
        
    def zero_forces(self):
        self.zero_F = float(self.force_val_raw)
        logger.info(f'Setting zero at {self.force_val_raw} N')

    def zero_elongation(self):
        return

    def stage_setup(self):
        pop = Popup(self,initial = {"gearbox": self.gear_ratio, "loadcell":self.loadcell,"calibration":self.calibration_factor,"force_lim":self.force_lim,"sampling":self.tensile_update_interval})
        pop.config_applied.connect(self.apply_config)  # connect signal -> slot
        pop.exec() 
        
    def apply_config(self,cfg: dict):
        self.current_config = cfg
        
        self.gear_ratio = cfg["gearbox"]
        self.loadcell = cfg["loadcell"]
        try:
            self.calibration_factor = float(cfg["calibration"])
        except ValueError as e:
            logger.error(f"failed to update calibration factor: {e}")
            return
        try:
            self.force_lim = float(cfg["force_lim"])
        except ValueError as e:
            logger.error(f"failed to update force limit: {e}")
            return
        try:
            self.tensile_update_interval = int(cfg["sampling"])
        except ValueError as e:
            logger.error(f"failed to update sampling time: {e}")
            return
        
        logger.info(f"Applied configuration:\ngear box: {self.gear_ratio},\nload cell: {self.loadcell},\ncalibration factor: {self.calibration_factor},\nforce limit: {self.force_lim} N,\nsampling interval {self.tensile_update_interval} ms")

    def _alert_window(self):
        """Raise window to front and flash taskbar for urgent alerts."""
        try:
            self.raise_()
            self.activateWindow()
            QApplication.alert(self, 0)  # flash until focused
        except Exception:
            pass

    def log_status_update(self):
        lines = ["--- 5-min Status Update ---"]
        for index, gas in enumerate(gas_types):
            if connect_status[index]:
                try:
                    flow = readings_dict[gas]["Flow rate [SLPM]"].text()
                    lines.append(f"  {gas} flow: {flow} SLPM")
                except Exception:
                    lines.append(f"  {gas} flow: N/A")
            else:
                lines.append(f"  {gas}: not connected")
        try:
            temp = self.temp_widget.temp_val
            if temp is not None:
                lines.append(f"  Temperature: {temp:.2f} °C")
            else:
                lines.append(f"  Temperature: no data (thermocouple not connected?)")
        except Exception:
            lines.append(f"  Temperature: N/A")
        logger.info("\n".join(lines))

    def closeEvent(self, event):
        reply = QMessageBox.question(self, "Warning", "Close Program?",
                                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply != QMessageBox.StandardButton.Yes:
            event.ignore(); return
        # Clean up temperature widget resources
        try:
            self.temp_widget.cleanup()
        except Exception:
            pass
        event.accept()
        sys.exit(0)

    # -----------------------
    # Worker to run tensile commands in background (non-blocking)
    # -----------------------
class TensileWorker(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, conn, cmds):
        super().__init__()
        self.conn = conn
        # ensure commands are bytes
        self.cmds = [c if isinstance(c, (bytes, bytearray)) else str(c).encode('ascii') for c in cmds]
        self._running = False

    def run(self):
        try:
            # send all setup commands once (Arduino holds state)
            with serial_lock:
                for cmd in self.cmds:
                    try:
                        self.conn.write(cmd)
                    except Exception as e:
                        logger.debug(f"TensileWorker write error: {e}")

            # mark running and wait until stop() is called
            self._running = True
            while self._running:
                QThread.msleep(100)  # low-frequency sleep to avoid busy loop
        except Exception as e:
            self.error.emit(str(e))
        finally:
            self.finished.emit()

    def stop(self):
        self._running = False

class Popup(QDialog):
    config_applied = pyqtSignal(dict)
    
    def __init__(self, parent=None, initial: dict | None = None):
        super().__init__(parent)
        self.setWindowTitle("Config")
        
        f = self.font()             
        f.setPointSize(12)           
        self.setFont(f)
        
        self.layout_main = QVBoxLayout(self)        
        ctrl_popup_layout = QGridLayout()
        self.layout_main.addLayout(ctrl_popup_layout)
    
        ctrl_popup_layout.addWidget(QLabel("Gearbox"), 0, 0)
        self.gearbox_box = QComboBox()
        self.gearbox_box.addItems(["None", "50:1"])
        ctrl_popup_layout.addWidget(self.gearbox_box, 0, 1)
        
        ctrl_popup_layout.addWidget(QLabel("Load cell type"), 1, 0)
        self.loadcell_box = QComboBox()
        self.loadcell_box.addItems(["2.5 kN", "50 kN"])
        self.loadcell_box.currentTextChanged.connect(self.text_changed)
        ctrl_popup_layout.addWidget(self.loadcell_box, 1, 1)
        
        # --------Separate layout for load cell settings --------
        loadcell_box_1 = QWidget()
        loadcell_box_1.setStyleSheet("font-size: 12pt;")
        loadcell_1_layout = QHBoxLayout()
        
        loadcell_1_layout.addWidget(QLabel("Calibration factor"))
        self.calibration_factor = QLineEdit()
        self.calibration_factor.setText("2715.0") 
        loadcell_1_layout.addWidget(self.calibration_factor)
        
        loadcell_1_layout.addWidget(QLabel("Force limit [N]"))
        self.force_limit = QLineEdit()
        self.force_limit.setText("2000.0")
        loadcell_1_layout.addWidget(self.force_limit)
        
        loadcell_box_1.setLayout(loadcell_1_layout)
        ctrl_popup_layout.addWidget(loadcell_box_1,2,1)
        # -------------------------------------------------------
        
        ctrl_popup_layout.addWidget(QLabel("Data sampling interval [ms]"), 4, 0)
        self.sampling_time = QLineEdit()
        self.sampling_time.setText("100")
        ctrl_popup_layout.addWidget(self.sampling_time, 4, 1)        
        
        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(self.on_accept)
        ctrl_popup_layout.addWidget(apply_btn,5,0)  
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.cancel)
        ctrl_popup_layout.addWidget(cancel_btn,5,1)  
        
        if initial: #load default values
            gear_ratio = initial["gearbox"]
            loadcell = initial["loadcell"]            
            
            gearbox_index = self.gearbox_box.findText(gear_ratio, Qt.MatchFlag.MatchFixedString) #sets gearbox dropdown based on index
            self.gearbox_box.setCurrentIndex(gearbox_index) if gearbox_index >= 0 else 0
            
            loadcell_index = self.loadcell_box.findText(loadcell, Qt.MatchFlag.MatchFixedString) #sets gearbox dropdown based on index
            self.loadcell_box.setCurrentIndex(loadcell_index) if loadcell_index >= 0 else 0
            
            self.calibration_factor.setText(str(initial["calibration"])) 
            self.force_limit.setText(str(initial["force_lim"]))
            self.sampling_time.setText(str(initial["sampling"]))
       
    def text_changed(self):
        if self.loadcell_box.currentText() == "2.5 kN":
            self.calibration_factor.setText("2715.0")
            self.force_limit.setText("2000.0")
        else:
            self.calibration_factor.setText("0")
            self.force_limit.setText("45000.0")        
       
    def on_accept(self):
        cfg = {
            "gearbox": self.gearbox_box.currentText(),
            "loadcell": self.loadcell_box.currentText(),
            "calibration": self.calibration_factor.text(),
            "force_lim": self.force_limit.text(),
            "sampling": self.sampling_time.text()
        }
        self.config_applied.emit(cfg)
        self.accept()
        
    def cancel(self):
        self.close()

# -----------------------
# Main entry point
# -----------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec())