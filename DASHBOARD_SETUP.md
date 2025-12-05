# Live Dashboard Setup

The live dashboard provides a web-based interface that can be opened in a browser on a separate display to monitor eye tracking tests in real-time.

## Installation

Install the required dependencies:

```bash
pip3 install flask flask-socketio
```

## Usage

1. Start the main eye tracking application:
   ```bash
   python3 main.py
   ```

2. The dashboard server will automatically start on port 5000.

3. Open a web browser on your second display and navigate to:
   ```
   http://localhost:5000
   ```

   Or if accessing from another device on the same network:
   ```
   http://[your-computer-ip]:5000
   ```

## Features

The dashboard displays:
- **Eye Camera Feed** (left): Live video feed from the eye tracking camera
- **Live Gaze Path** (right): Real-time visualization of gaze trajectory
- **Test Metrics** (bottom): Live metrics that update as tests progress:
  - Saccade Test: latency, velocity, accuracy, total saccades
  - Smooth Pursuit Test: gain, total measurements
  - Fixed Point Stability Test: deviation, total measurements
  - PLR Test: pupil diameter, total samples

## Layout

- Camera feed on the left
- Gaze path visualization on the right
- Metrics cards below both panels

The dashboard automatically updates when tests start/stop and shows real-time data as tests progress.

