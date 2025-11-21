# Eye Tracking Phygital

A real-time eye tracking application that detects pupil position and maps gaze to screen coordinates. The application uses computer vision techniques to track eye movement from video input (webcam or video file) and provides a visual overlay showing where you're looking on screen.

## Features

- **Real-time pupil detection** using ellipse fitting and contour analysis
- **Gaze mapping** to screen coordinates with geometric projection
- **Visual gaze overlay** showing a red dot indicating where you're looking
- **Calibration system** for accurate gaze tracking
- **Multiple input sources**: webcam or video file (MP4)
- **Debug mode** for visualizing the detection process

## Requirements

- macOS (tested on macOS 12+)
- Python 3.7 or higher
- Webcam (for live tracking) or MP4 video file
- Homebrew (for optional dependencies)

## Installation

### Quick Setup (Recommended)

1. **Clone or download this repository**

2. **Make the setup script executable and run it:**
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

   This script will:
   - Check for Python 3 and install it if needed
   - Install Homebrew if not present
   - Install ffmpeg for video codec support
   - Create a Python virtual environment
   - Install all required Python packages
   - Create a `videos` directory for input/output files
   - Set up a `.gitignore` file

### Manual Setup

If you prefer to set up manually:

1. **Install Python 3** (if not already installed):
   ```bash
   brew install python
   ```

2. **Install ffmpeg** (for video codec support):
   ```bash
   brew install ffmpeg
   ```

3. **Create a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

4. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

5. **Create videos directory:**
   ```bash
   mkdir -p videos
   ```

## Usage

### Activating the Virtual Environment

Before running the application, activate the virtual environment:

```bash
source venv/bin/activate
```

### Running the Application

Simply run:

```bash
python EyeTracking.py
```

The application will:
1. Attempt to open a hardcoded video path (if it exists)
2. If not found, prompt you to select a video file
3. Alternatively, you can modify the code to use webcam input directly

### Input Methods

The application supports two input methods:

1. **Video File**: Process a pre-recorded MP4 video file
2. **Webcam**: Real-time tracking from your camera (default in current code)

To switch between methods, modify line 936 in `EyeTracking.py`:
- `process_video(video_path, 1)` - Video file input
- `process_video(video_path, 2)` - Webcam input

### Controls

While the application is running:

- **SPACEBAR**: Pause/resume video playback
- **D**: Toggle debug mode (shows intermediate processing steps)
- **C**: Calibrate gaze tracking (look at screen center and press 'c')
- **G**: Toggle gaze overlay on/off
- **Q**: Quit the application

### Calibration

For accurate gaze tracking:

1. Position yourself comfortably in front of the camera
2. Look at the center of your screen
3. Press **'c'** to calibrate
4. The system will compute an offset to center your gaze

### Video Files

- **Input videos**: Place MP4 files in the `videos/` directory
- **Output video**: The processed video with overlays will be saved as `output_video.mp4` in the project root

## Project Structure

```
EyeTrackingPhygital/
├── EyeTracking.py          # Main application file
├── requirements.txt        # Python dependencies
├── setup.sh               # Setup script
├── README.md              # This file
├── venv/                  # Virtual environment (created by setup)
└── videos/                # Directory for input/output videos (created by setup)
```

## Dependencies

- **opencv-python**: Computer vision library for image processing
- **numpy**: Numerical computing
- **matplotlib**: Plotting and visualization
- **tkinter**: GUI components (comes with Python)

## Configuration

You can adjust these parameters in `EyeTracking.py`:

- `DISPLAY_DISTANCE_CM` (line 14): Distance from eye to display in centimeters
- `DISPLAY_DPI` (line 16): Your monitor's DPI (typical: 96, high-DPI: 144-192)

## Troubleshooting

### Camera Not Working

- Ensure your camera permissions are enabled in System Preferences
- Try a different camera index if you have multiple cameras
- Check that no other application is using the camera

### Video Codec Issues

- Ensure ffmpeg is installed: `brew install ffmpeg`
- Try converting your video to a standard MP4 format

### Gaze Overlay Not Appearing

- Press **'g'** to toggle the overlay
- Check that the overlay window isn't hidden behind other windows
- On macOS, you may need to grant accessibility permissions

### Poor Tracking Accuracy

- Ensure good lighting conditions
- Calibrate by looking at screen center and pressing **'c'**
- Adjust `DISPLAY_DISTANCE_CM` to match your setup
- Try enabling debug mode (**'d'**) to see detection quality

## Development

### Adding New Features

The code is organized into several sections:
- Image processing functions (thresholding, contour detection)
- Ellipse fitting and validation
- Gaze computation and calibration
- Overlay window management
- Video processing loop

### Debug Mode

Enable debug mode with **'d'** to see:
- Thresholded images at different levels
- Contour detection results
- Ellipse fitting quality metrics
- Intermediate processing steps

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]

## Acknowledgments

This project uses OpenCV for computer vision processing and tkinter for GUI components.

