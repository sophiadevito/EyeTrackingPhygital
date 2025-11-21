#!/bin/bash

# Eye Tracking Phygital Project Setup Script
# This script sets up the development environment for the Eye Tracking project

set -e  # Exit on any error

echo "=========================================="
echo "Eye Tracking Phygital - Setup Script"
echo "=========================================="
echo ""

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check for Python 3
if ! command_exists python3; then
    echo "❌ Python 3 not found!"
    echo "Installing Python 3 via Homebrew..."
    
    # Check for Homebrew first
    if ! command_exists brew; then
        echo "❌ Homebrew not found. Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        
        # Add Homebrew to PATH for Apple Silicon Macs
        if [[ $(uname -m) == "arm64" ]]; then
            echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
            eval "$(/opt/homebrew/bin/brew shellenv)"
        fi
    fi
    
    brew install python
else
    echo "✅ Python 3 found: $(python3 --version)"
fi

# Check for Homebrew (needed for optional dependencies)
if ! command_exists brew; then
    echo "⚠️  Homebrew not found. Some optional dependencies may not be available."
    echo "   You can install it later with: /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
else
    echo "✅ Homebrew found"
fi

# Install ffmpeg for better video codec support (optional but recommended)
if command_exists brew; then
    if ! command_exists ffmpeg; then
        echo "📦 Installing ffmpeg for video codec support..."
        brew install ffmpeg
    else
        echo "✅ ffmpeg already installed: $(ffmpeg -version | head -n 1)"
    fi
else
    echo "⚠️  Skipping ffmpeg installation (Homebrew not available)"
    echo "   You can install it manually later if needed for video processing"
fi

# Create virtual environment
if [ ! -d "venv" ]; then
    echo ""
    echo "📦 Creating Python virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip --quiet

# Install Python dependencies
echo ""
echo "📦 Installing Python dependencies from requirements.txt..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo "✅ Dependencies installed"
else
    echo "❌ Error: requirements.txt not found!"
    exit 1
fi

# Create videos directory for input/output videos
if [ ! -d "videos" ]; then
    echo ""
    echo "📁 Creating videos directory for input/output video files..."
    mkdir -p videos
    echo "✅ Videos directory created"
    echo "   Place your input MP4 files in the 'videos' directory"
else
    echo "✅ Videos directory already exists"
fi

# Create .gitignore if it doesn't exist
if [ ! -f ".gitignore" ]; then
    echo ""
    echo "📝 Creating .gitignore file..."
    cat > .gitignore << EOF
# Python
venv/
__pycache__/
*.pyc
*.pyo
*.pyd
.Python

# Video files
videos/*.mp4
output_video.mp4

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
EOF
    echo "✅ .gitignore created"
fi

echo ""
echo "=========================================="
echo "✅ Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Place your input video files in the 'videos' directory"
echo ""
echo "3. Run the application:"
echo "   python EyeTracking.py"
echo ""
echo "Note: The application will create 'output_video.mp4' in the project root"
echo "      when processing video files."
echo ""

