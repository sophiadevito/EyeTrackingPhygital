"""
Live Dashboard Server for Eye Tracking Tests

Provides a web-based dashboard that can be opened in a browser on a separate display.
Shows live camera feed, gaze path, and real-time test metrics.
"""

from flask import Flask, Response, render_template_string, send_from_directory
from flask_socketio import SocketIO, emit
import cv2
import base64
import threading
import time
import json
import os
from datetime import datetime

app = Flask(__name__)
app.config['SECRET_KEY'] = 'rumble_rims_dashboard_secret'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global state
dashboard_active = False
camera_frame = None
camera_lock = threading.Lock()
current_test_data = {
    'test_name': None,
    'test_active': False,
    'gaze_path': [],
    'metrics': {},
    'targets': []
}

# Dashboard HTML template
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Rumble Rims Live Dashboard</title>
    <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Epilogue', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1E2D59;
            color: white;
            padding: 20px;
            overflow-x: hidden;
        }
        
        .header-container {
            display: grid;
            grid-template-columns: 1fr 2fr;
            gap: 20px;
            margin-bottom: 20px;
        }
        
        .header {
            background: #9E1B32;
            padding: 20px;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 20px;
            border-radius: 8px;
        }
        
        .header .logo {
            height: 60px;
            width: auto;
            object-fit: contain;
        }
        
        .header h1 {
            font-size: 28px;
            margin: 0;
        }
        
        .main-container {
            display: grid;
            grid-template-columns: 1fr 2fr;
            gap: 20px;
            margin-bottom: 20px;
        }
        
        .panel {
            background: #2a3d6b;
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        }
        
        .panel h2 {
            font-size: 18px;
            margin-bottom: 15px;
            color: white;
            border-bottom: 2px solid #9E1B32;
            padding-bottom: 8px;
        }
        
        .camera-container {
            position: relative;
            width: 100%;
            background: #000;
            border-radius: 4px;
            overflow: hidden;
        }
        
        .camera-feed {
            width: 100%;
            height: auto;
            display: block;
        }
        
        .gaze-view-container {
            position: relative;
            width: 100%;
            height: 400px;
            background: #f8f9fa;
            border-radius: 4px;
            overflow: hidden;
        }
        
        .gaze-canvas {
            width: 100%;
            height: 100%;
        }
        
        .test-status-panel {
            background: #2a3d6b;
            border-radius: 8px;
            padding: 30px 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
            display: flex;
            align-items: center;
            justify-content: center;
            transition: background 0.3s ease;
        }
        
        .test-status-panel.active {
            background: #28a745;
        }
        
        .test-status-panel.inactive {
            background: #6c757d;
        }
        
        .test-status-panel.calibrating {
            background: #ffc107;
        }
        
        .test-status-large {
            font-size: 42px;
            font-weight: 700;
            color: white;
            text-align: center;
            margin: 0;
            line-height: 1.2;
        }
        
        .test-status-indicator-large {
            display: inline-block;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            margin-right: 15px;
            vertical-align: middle;
        }
        
        .test-status-indicator-large.active {
            background: white;
            animation: pulse 2s infinite;
        }
        
        .test-status-indicator-large.inactive {
            background: #adb5bd;
        }
        
        .metrics-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }
        
        .metric-card {
            background: #3a4d7b;
            padding: 15px;
            border-radius: 6px;
            border-left: 4px solid #667eea;
        }
        
        .metric-label {
            font-size: 12px;
            color: #adb5bd;
            margin-bottom: 5px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .metric-value {
            font-size: 24px;
            font-weight: 600;
            color: white;
        }
        
        .metric-unit {
            font-size: 14px;
            color: #adb5bd;
            margin-left: 5px;
        }
        
        .no-data {
            color: #6c757d;
            font-style: italic;
            text-align: center;
            padding: 40px;
        }
        
        .test-active {
            background: #28a745;
            animation: pulse 2s infinite;
        }
        
        .test-inactive {
            background: #6c757d;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
    </style>
</head>
<body>
    <div class="header-container">
        <div class="header">
            <img src="/ram_logo.png" alt="Rumble Rims Logo" class="logo">
            <div>
                <h1>Rumble Rims Live Dashboard</h1>
            </div>
        </div>
        
        <div class="test-status-panel inactive" id="testStatusPanel">
            <p class="test-status-large">
                <span class="test-status-indicator-large inactive" id="testIndicatorLarge"></span>
                <span id="testStatusLarge">No test active</span>
            </p>
        </div>
    </div>
    
    <div class="main-container">
        <div class="panel">
            <h2>Eye Camera Feed</h2>
            <div class="camera-container">
                <img id="cameraFeed" class="camera-feed" src="/video_feed" alt="Camera Feed">
            </div>
        </div>
        
        <div class="panel">
            <h2>Live Gaze Path</h2>
            <div class="gaze-view-container">
                <canvas id="gazeCanvas" class="gaze-canvas"></canvas>
            </div>
        </div>
    </div>
    
    <div class="panel" style="margin-top: 20px;">
        <h2>Test Metrics</h2>
        <div class="metrics-container" id="metricsContainer">
            <div class="no-data">Waiting for test data...</div>
        </div>
    </div>
    
    <script>
        const socket = io();
        const canvas = document.getElementById('gazeCanvas');
        const ctx = canvas.getContext('2d');
        const testStatusPanel = document.getElementById('testStatusPanel');
        const testIndicatorLarge = document.getElementById('testIndicatorLarge');
        const testStatusLarge = document.getElementById('testStatusLarge');
        const metricsContainer = document.getElementById('metricsContainer');
        
        // Set canvas size
        function resizeCanvas() {
            const container = canvas.parentElement;
            canvas.width = container.clientWidth;
            canvas.height = container.clientHeight;
        }
        resizeCanvas();
        window.addEventListener('resize', resizeCanvas);
        
        // Gaze path storage
        let gazePath = [];
        let targets = [];
        let monitorWidth = 1920;
        let monitorHeight = 1080;
        
        // Draw gaze visualization
        function drawGazeView() {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            if (gazePath.length < 2) {
                ctx.fillStyle = '#6c757d';
                ctx.font = '16px Epilogue';
                ctx.textAlign = 'center';
                ctx.fillText('No gaze data yet', canvas.width / 2, canvas.height / 2);
                return;
            }
            
            // Calculate bounds
            const xs = gazePath.map(p => p.x);
            const ys = gazePath.map(p => p.y);
            const minX = Math.min(...xs);
            const maxX = Math.max(...xs);
            const minY = Math.min(...ys);
            const maxY = Math.max(...ys);
            
            const padding = 50;
            const rangeX = maxX - minX || 1;
            const rangeY = maxY - minY || 1;
            
            const scaleX = (canvas.width - padding * 2) / rangeX;
            const scaleY = (canvas.height - padding * 2) / rangeY;
            const scale = Math.min(scaleX, scaleY);
            
            const offsetX = (canvas.width - rangeX * scale) / 2 - minX * scale;
            const offsetY = (canvas.height - rangeY * scale) / 2 - minY * scale;
            
            // Draw grid
            ctx.strokeStyle = '#e9ecef';
            ctx.lineWidth = 1;
            for (let i = 0; i <= 5; i++) {
                const x = padding + (canvas.width - padding * 2) * (i / 5);
                const y = padding + (canvas.height - padding * 2) * (i / 5);
                ctx.beginPath();
                ctx.moveTo(x, padding);
                ctx.lineTo(x, canvas.height - padding);
                ctx.stroke();
                ctx.beginPath();
                ctx.moveTo(padding, y);
                ctx.lineTo(canvas.width - padding, y);
                ctx.stroke();
            }
            
            // Draw targets
            ctx.fillStyle = '#28a745';
            targets.forEach(target => {
                const x = target.x * scale + offsetX;
                const y = target.y * scale + offsetY;
                ctx.beginPath();
                ctx.arc(x, y, 8, 0, Math.PI * 2);
                ctx.fill();
            });
            
            // Draw gaze path
            if (gazePath.length > 1) {
                ctx.strokeStyle = '#667eea';
                ctx.lineWidth = 2;
                ctx.beginPath();
                ctx.moveTo(gazePath[0].x * scale + offsetX, gazePath[0].y * scale + offsetY);
                for (let i = 1; i < gazePath.length; i++) {
                    ctx.lineTo(gazePath[i].x * scale + offsetX, gazePath[i].y * scale + offsetY);
                }
                ctx.stroke();
                
                // Draw current position
                if (gazePath.length > 0) {
                    const last = gazePath[gazePath.length - 1];
                    ctx.fillStyle = '#dc3545';
                    ctx.beginPath();
                    ctx.arc(last.x * scale + offsetX, last.y * scale + offsetY, 6, 0, Math.PI * 2);
                    ctx.fill();
                }
            }
        }
        
        // Socket event handlers
        socket.on('connect', () => {
            console.log('Connected to dashboard server');
        });
        
        socket.on('gaze_update', (data) => {
            if (data.x !== null && data.y !== null) {
                gazePath.push({ x: data.x, y: data.y, timestamp: data.timestamp });
                // Keep only last 1000 points
                if (gazePath.length > 1000) {
                    gazePath.shift();
                }
                drawGazeView();
            }
        });
        
        socket.on('test_update', (data) => {
            if (data.is_calibrating) {
                testStatusPanel.className = 'test-status-panel calibrating';
                testIndicatorLarge.className = 'test-status-indicator-large inactive';
                testStatusLarge.textContent = 'Recalibrating...';
            } else if (data.test_active) {
                testStatusPanel.className = 'test-status-panel active';
                testIndicatorLarge.className = 'test-status-indicator-large active';
                testStatusLarge.textContent = data.test_name || 'Unknown Test';
            } else {
                testStatusPanel.className = 'test-status-panel inactive';
                testIndicatorLarge.className = 'test-status-indicator-large inactive';
                testStatusLarge.textContent = 'No test active';
            }
            
            if (data.metrics) {
                updateMetrics(data.metrics);
            }
            
            if (data.targets) {
                targets = data.targets;
                drawGazeView();
            }
            
            if (data.monitor_width && data.monitor_height) {
                monitorWidth = data.monitor_width;
                monitorHeight = data.monitor_height;
            }
        });
        
        socket.on('gaze_path_reset', () => {
            gazePath = [];
            targets = [];
            drawGazeView();
        });
        
        function updateMetrics(metrics) {
            if (!metrics || Object.keys(metrics).length === 0) {
                metricsContainer.innerHTML = '<div class="no-data">Waiting for test data...</div>';
                return;
            }
            
            const metricCards = Object.entries(metrics).map(([key, value]) => {
                const label = key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
                const formattedValue = typeof value === 'number' ? value.toFixed(2) : value;
                const unit = getUnit(key);
                
                return `
                    <div class="metric-card">
                        <div class="metric-label">${label}</div>
                        <div class="metric-value">
                            ${formattedValue}<span class="metric-unit">${unit}</span>
                        </div>
                    </div>
                `;
            }).join('');
            
            metricsContainer.innerHTML = metricCards;
        }
        
        function getUnit(key) {
            if (key.includes('latency') || key.includes('ms')) return 'ms';
            if (key.includes('velocity')) return '°/ms';
            if (key.includes('accuracy') || key.includes('percent') || key.includes('rate')) return '%';
            if (key.includes('deviation') || key.includes('degrees')) return '°';
            if (key.includes('gain')) return '';
            if (key.includes('count') || key.includes('trials')) return '';
            return '';
        }
        
        // Initial draw
        drawGazeView();
    </script>
</body>
</html>
"""

def update_camera_frame(frame):
    """Update the camera frame for streaming"""
    global camera_frame
    with camera_lock:
        camera_frame = frame.copy() if frame is not None else None

def update_gaze(gaze_x, gaze_y):
    """Send gaze update to dashboard"""
    if dashboard_active:
        socketio.emit('gaze_update', {
            'x': gaze_x,
            'y': gaze_y,
            'timestamp': time.time()
        })

def update_test_status(test_name, test_active, metrics=None, targets=None, monitor_width=None, monitor_height=None, is_calibrating=False):
    """Update test status and metrics"""
    global current_test_data
    current_test_data['test_name'] = test_name
    current_test_data['test_active'] = test_active
    if metrics:
        current_test_data['metrics'] = metrics
    if targets:
        current_test_data['targets'] = targets
    
    if dashboard_active:
        socketio.emit('test_update', {
            'test_name': test_name,
            'test_active': test_active,
            'metrics': metrics or {},
            'targets': targets or [],
            'monitor_width': monitor_width,
            'monitor_height': monitor_height,
            'is_calibrating': is_calibrating
        })

def reset_gaze_path():
    """Reset the gaze path visualization"""
    if dashboard_active:
        socketio.emit('gaze_path_reset')

@app.route('/')
def index():
    """Serve the dashboard page"""
    return render_template_string(DASHBOARD_HTML)

def generate_frames():
    """Generate MJPEG frames from camera"""
    global camera_frame
    while True:
        with camera_lock:
            if camera_frame is not None:
                ret, buffer = cv2.imencode('.jpg', camera_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ret:
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            else:
                # Send a blank frame if no camera data
                blank_frame = cv2.zeros((480, 640, 3), dtype=cv2.uint8)
                cv2.putText(blank_frame, 'No camera feed', (200, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                ret, buffer = cv2.imencode('.jpg', blank_frame)
                if ret:
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.033)  # ~30 FPS

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/ram_logo.png')
def ram_logo():
    """Serve the ram logo image"""
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), 'ram_logo.png')

def start_dashboard(port=5000, debug=False):
    """Start the dashboard server"""
    global dashboard_active
    dashboard_active = True
    print(f"\n{'='*60}")
    print("Starting Live Dashboard Server")
    print(f"{'='*60}")
    print(f"Dashboard available at: http://localhost:{port}")
    print(f"Open this URL in a browser on your second display")
    print(f"Press Ctrl+C to stop the dashboard")
    print(f"{'='*60}\n")
    
    try:
        socketio.run(app, host='0.0.0.0', port=port, debug=debug, allow_unsafe_werkzeug=True)
    except KeyboardInterrupt:
        print("\nDashboard server stopped.")
        dashboard_active = False

def stop_dashboard():
    """Stop the dashboard server"""
    global dashboard_active
    dashboard_active = False

if __name__ == '__main__':
    start_dashboard()

