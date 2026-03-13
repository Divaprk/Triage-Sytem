"""
Raspberry Pi Vitals Monitor
- MAX30100: Heart Rate + SpO2 (ratio-of-ratios algorithm)
- MLX90614: Non-contact IR Temperature (object/ambient)
- Optional: POST to Windows endpoint
"""

import time
import requests
import numpy as np
import os
import psutil
import smbus2  # New import for MLX90614
from max30100 import MAX30100
import json
from collections import deque

# ----------------------------
# CONFIG
# ----------------------------
WINDOWS_IP = "192.168.137.1"
ENDPOINT = f"http://{WINDOWS_IP}:5000/update_vitals"
ENABLE_HTTP_POST = False

SAMPLE_RATE = 0.05             # 20 Hz sampling
WINDOW_SECONDS = 8
MIN_INTERVAL = 0.4
MAX_BPM = 160
MIN_BPM = 40

# SpO2 algorithm parameters
SPO2_WINDOW = 32
MIN_DC_IR = 500
MIN_DC_RED_RATIO = 0.25
MIN_AC_DC_RATIO = 0.015
SPO2_LOW_THRESHOLD = 85
SPO2_HIGH_THRESHOLD = 100

# Finger detection thresholds
MIN_DC_IR_WITH_FINGER = 500
MAX_DC_IR_NO_FINGER = 300
MIN_AC_DC_RATIO_FOR_PULSE = 0.004
MIN_PEAKS_REQUIRED = 2
PEAK_CONSISTENCY_TOLERANCE = 0.6

# Buffer management
MAX_BUFFER_SIZE = 200

# Debug mode
DEBUG = False

# MLX90614 Configuration
MLX90614_ADDR = 0x5A
MLX90614_REG_AMBIENT = 0x06
MLX90614_REG_OBJECT = 0x07
MLX90614_I2C_BUS = 1  # Standard on Raspberry Pi
USE_OBJECT_TEMP = True  # Set False to use ambient temperature instead

TEMP_CALIBRATION_OFFSET = +2.0

# ----------------------------
# INIT SENSORS
# ----------------------------
print("Initializing sensors...")

# MLX90614 IR Temperature Sensor
def init_mlx90614():
    """Initialize I2C bus for MLX90614"""
    try:
        bus = smbus2.SMBus(MLX90614_I2C_BUS)
        print("? MLX90614 initialized")
        return bus
    except Exception as e:
        print(f"? MLX90614 init failed: {e}")
        return None

mlx_bus = init_mlx90614()

def read_mlx90614_temperature(bus, register):
    """Read temperature from MLX90614 with calibration offset."""
    if bus is None:
        return None
    try:
        data = bus.read_i2c_block_data(MLX90614_ADDR, register, 3)
        raw_temp = data[0] | (data[1] << 8)
        temp_c = raw_temp * 0.02 - 273.15
        
        # ?? Apply calibration offset
        temp_c += TEMP_CALIBRATION_OFFSET
        
        return round(temp_c, 2)
    except Exception as e:
        print(f"? MLX90614 read error: {e}")
        return None

# MAX30100 Pulse Oximeter
m = MAX30100()
m.enable_spo2()


# Sliding window buffers
ir_buffer = deque(maxlen=MAX_BUFFER_SIZE)
red_buffer = deque(maxlen=MAX_BUFFER_SIZE)
time_buffer = deque(maxlen=MAX_BUFFER_SIZE)

# State tracking
last_valid_spo2 = None
last_valid_hr = None
consecutive_failures = 0

print("? Sensors initialized. Starting vitals monitor (EDGE processing enabled)...")
system_start = time.time()

# ----------------------------
# Helper Functions
# ----------------------------

def detect_finger_and_signal_quality(ir_buf, red_buf):
    """Detect if finger is present AND signal is good enough for vitals."""
    if len(ir_buf) < 20 or len(red_buf) < 20:
        return False, False, "insufficient_samples"
    
    ir_recent = np.array(list(ir_buf)[-40:])
    red_recent = np.array(list(red_buf)[-40:])
    
    dc_ir = np.mean(ir_recent)
    dc_red = np.mean(red_recent)
    
    if dc_ir < MIN_DC_IR_WITH_FINGER:
        return False, False, f"no_finger(dc_ir={dc_ir:.0f}<{MIN_DC_IR_WITH_FINGER})"
    
    if dc_red < dc_ir * 0.2:
        return False, False, f"red_led_weak(dc_red={dc_red:.0f})"
    
    ac_ir = np.std(ir_recent)
    ac_red = np.std(red_recent)
    
    if dc_ir > 0:
        ac_dc_ratio = ac_ir / dc_ir
        if ac_dc_ratio < MIN_AC_DC_RATIO_FOR_PULSE:
            return True, False, f"no_pulse(ac_dc={ac_dc_ratio:.3f}<{MIN_AC_DC_RATIO_FOR_PULSE})"
    
    if ac_ir < 10:
        return True, False, f"signal_too_weak(ac_ir={ac_ir:.1f})"
    
    return True, True, "ok"
    
    
def calculate_spo2_ratio(red_buf, ir_buf, window=SPO2_WINDOW):
    """Improved SpO2 calculation with better filtering"""
    if len(red_buf) < window or len(ir_buf) < window:
        return None, "insufficient_data"
    
    red = np.array(list(red_buf)[-window:])
    ir = np.array(list(ir_buf)[-window:])
    
    # ---- Improved DC removal: moving average ----
    # Use a longer window for DC to avoid pulse contamination
    dc_window = max(8, window // 2)
    dc_red = np.convolve(red, np.ones(dc_window)/dc_window, mode='same')[-window:]
    dc_ir = np.convolve(ir, np.ones(dc_window)/dc_window, mode='same')[-window:]
    
    # AC = signal minus DC (pulsatile component only)
    ac_red_signal = red - dc_red
    ac_ir_signal = ir - dc_ir
    
    # ---- Optional: Simple bandpass via high-pass + low-pass ----
    # High-pass: remove very slow drift (cutoff ~0.5 Hz)
    # Low-pass: remove high-frequency noise (cutoff ~5 Hz)
    # For 20 Hz sampling, a simple 3-point smoother works as low-pass:
    ac_red_signal = np.convolve(ac_red_signal, [0.25, 0.5, 0.25], mode='same')
    ac_ir_signal = np.convolve(ac_ir_signal, [0.25, 0.5, 0.25], mode='same')
    
    # RMS of AC components
    ac_red = np.sqrt(np.mean(ac_red_signal ** 2))
    ac_ir = np.sqrt(np.mean(ac_ir_signal ** 2))
    
    # DC values for ratio (use mean of DC estimates)
    dc_red_mean = np.mean(dc_red)
    dc_ir_mean = np.mean(dc_ir)
    
    if dc_ir_mean < 1 or dc_red_mean < 1 or ac_ir < 0.1:
        return None, "weak_signal"
    
    # Ratio of ratios
    R = (ac_red / dc_red_mean) / (ac_ir / dc_ir_mean)
    
    # ---- Calibrated SpO2 mapping ----
    SPO2_COEF_A = 112
    SPO2_COEF_B = 18
    spo2 = SPO2_COEF_A - SPO2_COEF_B * R
    
    # Validation (unchanged)
    if SPO2_LOW_THRESHOLD <= spo2 <= SPO2_HIGH_THRESHOLD:
        return int(round(spo2)), "valid"
    elif 70 <= spo2 < SPO2_LOW_THRESHOLD:
        return int(round(spo2)), "low_confidence"
    
    return None, f"out_of_range({spo2:.1f})"


def validate_with_hysteresis(current_value, last_valid, min_valid, max_valid, hold_cycles=10):
    """Apply hysteresis to prevent flickering between valid/None."""
    global consecutive_failures
    
    if current_value is not None and min_valid <= current_value <= max_valid:
        consecutive_failures = 0
        return current_value
    
    if last_valid is not None and consecutive_failures < hold_cycles:
        consecutive_failures += 1
        return last_valid
    
    consecutive_failures = 0
    return None
    
def detect_heart_rate(time_buf, ir_buf, min_interval=MIN_INTERVAL, min_bpm=MIN_BPM, max_bpm=MAX_BPM):
    """Detect heart rate with strict validation to reject noise."""
    min_samples = int(4.0 / SAMPLE_RATE)
    if len(ir_buf) < min_samples or len(time_buf) < min_samples:
        return None, "insufficient_data"
    
    recent_window = int(5.0 / SAMPLE_RATE)
    ir_recent = np.array(list(ir_buf)[-recent_window:])
    time_recent = np.array(list(time_buf)[-recent_window:])
    
    if len(ir_recent) < 30:
        return None, "insufficient_recent_data"
    
    window = 9
    ir_filtered = np.convolve(ir_recent, np.ones(window)/window, mode='same')
    
    peaks = []
    mean_ir = np.mean(ir_filtered)
    std_ir = np.std(ir_filtered)
    threshold = mean_ir + 0.25 * std_ir
    
    for i in range(3, len(ir_filtered) - 3):
        is_peak = (ir_filtered[i] > ir_filtered[i-1] and 
                   ir_filtered[i] > ir_filtered[i-2] and
                   ir_filtered[i] > ir_filtered[i-3] and
                   ir_filtered[i] > ir_filtered[i+1] and
                   ir_filtered[i] > ir_filtered[i+2] and
                   ir_filtered[i] > ir_filtered[i+3])
        
        if is_peak and ir_filtered[i] > threshold:
            peak_amplitude = ir_filtered[i] - mean_ir
            if peak_amplitude > std_ir * 0.5:
                peaks.append(time_recent[i])
    
    filtered_peaks = []
    for t in peaks:
        if not filtered_peaks or (t - filtered_peaks[-1]) > min_interval:
            filtered_peaks.append(t)
    
    if len(filtered_peaks) < MIN_PEAKS_REQUIRED:
        return None, f"too_few_peaks({len(filtered_peaks)}<{MIN_PEAKS_REQUIRED})"
    
    intervals = np.diff(filtered_peaks)
    avg_interval = np.mean(intervals)
    std_interval = np.std(intervals)
    
    if avg_interval <= 0:
        return None, "invalid_intervals"
    
    if std_interval / avg_interval > PEAK_CONSISTENCY_TOLERANCE:
        return None, f"irregular_peaks(cv={std_interval/avg_interval:.2f})"
    
    bpm_candidate = 60 / avg_interval
    
    if min_bpm <= bpm_candidate <= max_bpm:
        return round(bpm_candidate, 1), "valid"
    
    return None, f"out_of_range({bpm_candidate:.1f})"


def send_to_endpoint(data):
    """Optional: POST vitals to Windows endpoint."""
    if not ENABLE_HTTP_POST:
        return
    
    try:
        response = requests.post(ENDPOINT, json=data, timeout=2)
        if response.status_code != 200:
            print(f"? HTTP POST failed: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"? HTTP connection error: {e}")
        
        
# ----------------------------
# MAIN LOOP
# ----------------------------
try:
    while True:
        now = time.time()
        
        # ---- Read MAX30100 ----
        try:
            m.read_sensor()
            ir_value = m.ir
            red_value = m.red
        except Exception as e:
            print(f"? MAX30100 read error: {e}")
            ir_value, red_value = None, None
        
        # ---- Read MLX90614 Temperature ----  UPDATED SECTION
        try:
            if USE_OBJECT_TEMP:
                temperature = read_mlx90614_temperature(mlx_bus, MLX90614_REG_OBJECT)
            else:
                temperature = read_mlx90614_temperature(mlx_bus, MLX90614_REG_AMBIENT)
        except Exception as e:
            print(f"? MLX90614 read error: {e}")
            temperature = None
        
        # ---- Store in sliding buffers ----
        if ir_value is not None and red_value is not None:
            ir_buffer.append(ir_value)
            red_buffer.append(red_value)
            time_buffer.append(now)
        
        # ---- Process when enough data accumulated ----
        if len(ir_buffer) >= int(WINDOW_SECONDS / SAMPLE_RATE):
            
            finger_present, signal_good, detection_reason = detect_finger_and_signal_quality(ir_buffer, red_buffer)
            
            if not finger_present:
                hr = None
                spo2 = None
                last_valid_hr = None
                last_valid_spo2 = None
                consecutive_failures = 0
                temp_str = f"{temperature:.1f}C" if temperature else "N/A"
                print(f"? NO FINGER DETECTED | HR: None | SpO2: None | Temp: {temp_str}")
                time.sleep(SAMPLE_RATE)
                continue
            
            if signal_good:
                hr, hr_confidence = detect_heart_rate(time_buffer, ir_buffer)
                hr = validate_with_hysteresis(hr, last_valid_hr, MIN_BPM, MAX_BPM, hold_cycles=2)
                if hr is not None:
                    last_valid_hr = hr
                
                spo2_raw, spo2_status = calculate_spo2_ratio(red_buffer, ir_buffer)
                
                if spo2_status == "valid":
                    spo2 = validate_with_hysteresis(spo2_raw, last_valid_spo2, SPO2_LOW_THRESHOLD, SPO2_HIGH_THRESHOLD)
                elif spo2_status == "low_confidence":
                    spo2 = validate_with_hysteresis(spo2_raw, last_valid_spo2, 70, 100)
                else:
                    spo2 = None
                
                if spo2 is not None:
                    last_valid_spo2 = spo2
            else:
                hr = None
                spo2 = None

            
            # Display output
            status_icon = "?" if signal_good else "?"
            temp_str = f"{temperature:.1f}C" if temperature is not None else "N/A"
            hr_str = f"{hr:.1f}" if hr is not None else "None"
            spo2_str = f"{spo2}%" if spo2 is not None else "None"
            
            print(f"{status_icon} HR: {hr_str} BPM | SpO2: {spo2_str} | Temp: {temp_str} [{detection_reason}]")
            
            # Optional HTTP POST
            if ENABLE_HTTP_POST:
                payload = {
                    "timestamp": time.time(),
                    "heart_rate_bpm": hr,
                    "spo2_percent": spo2,
                    "temperature_c": temperature,
                    "finger_present": finger_present,
                    "signal_quality": detection_reason
                }
                send_to_endpoint(payload)
        
        if DEBUG:
            ir_recent = np.array(list(ir_buffer)[-40:])
            print(f"  [CALIBRATE] DC_IR: {np.mean(ir_recent):.0f} | "
                  f"AC_IR: {np.std(ir_recent):.1f} | "
                  f"AC/DC: {np.std(ir_recent)/np.mean(ir_recent):.3f}")
        
        # Maintain sampling rate
        elapsed = time.time() - now
        sleep_time = max(0, SAMPLE_RATE - elapsed)
        time.sleep(sleep_time)
        
except KeyboardInterrupt:
    print("\n? Shutdown requested by user")

finally:
    print("Cleaning up sensors...")
    try:
        m.shutdown()
    except:
        pass
    # Close MLX90614 I2C bus UPDATED CLEANUP
    if mlx_bus:
        try:
            mlx_bus.close()
            print("? MLX90614 I2C bus closed")
        except:
            pass
    print("? Sensors shut down. Goodbye!")
