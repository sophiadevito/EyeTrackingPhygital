# Center Square Mask Module

The `mask.py` module provides masking and enhancement functionality for video processing in the eye tracking system.

## Purpose

Applies a mask that restricts processing to a 200x200 pixel square at a fixed position, with all pixels outside the square set to white (255). Optionally applies gamma correction and CLAHE enhancement to the square region to improve pupil detection. This is useful for:
- Focusing processing on a specific region of interest
- Reducing computational load
- Eliminating noise from other areas of the frame
- Improving contrast within the masked region for better pupil detection

## Configuration

Edit parameters in `mask.py`:

```python
# Center square mask parameters
USE_CENTER_SQUARE_MASK = True  # Enable/disable center square mask
CENTER_SQUARE_SIZE = 200  # Size of the square in pixels (200x200)
CENTER_SQUARE_X = 250  # X center position (None = frame center)
CENTER_SQUARE_Y = 300  # Y center position (None = frame center)

# Mask region enhancement parameters
USE_MASK_REGION_ENHANCEMENT = True  # Enable/disable enhancement of masked region
MASK_GAMMA_VALUE = 0.5  # Gamma value for masked region (lower = brighter dark areas)
MASK_CLAHE_CLIP_LIMIT = 3.0  # CLAHE clip limit for masked region
MASK_CLAHE_TILE_SIZE = (4, 4)  # CLAHE tile size for masked region (smaller = more adaptive)
```

## How It Works

1. **Applied After**: Grayscale conversion
2. **Applied Before**: Finding the darkest point
3. **Processing Order**:
   - Mask applied (pixels outside square become white)
   - Gamma correction + CLAHE applied to square region only
   - Darkest point detection
4. **Effect**: Only the square region is processed and enhanced

## Enhancement Details

### Gamma Correction
- Default: 0.5 (brightens dark areas like the pupil)
- Applied only to pixels within the square
- Uses lookup table for fast processing

### CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Default clip limit: 3.0
- Default tile size: 4x4 (more adaptive)
- Improves local contrast within the square
- Helps identify low-contrast pupils

## Usage

### Enable the Mask

Set in `mask.py`:
```python
USE_CENTER_SQUARE_MASK = True
```

### Adjust Square Size

```python
CENTER_SQUARE_SIZE = 300  # 300x300 pixel square
```

### Set Custom Center Position

```python
CENTER_SQUARE_X = 320  # X coordinate (in pixels)
CENTER_SQUARE_Y = 240  # Y coordinate (in pixels)
```

Leave as `None` to use the frame center:
```python
CENTER_SQUARE_X = None  # Uses frame center
CENTER_SQUARE_Y = None  # Uses frame center
```

## Example Configurations

### Default (Disabled)
```python
USE_CENTER_SQUARE_MASK = False
```

### Enable Both Mask and Enhancement
```python
USE_CENTER_SQUARE_MASK = True
USE_MASK_REGION_ENHANCEMENT = True
MASK_GAMMA_VALUE = 0.5
MASK_CLAHE_CLIP_LIMIT = 3.0
```

### Custom Enhancement Settings
```python
USE_MASK_REGION_ENHANCEMENT = True
MASK_GAMMA_VALUE = 0.4  # More aggressive brightening
MASK_CLAHE_CLIP_LIMIT = 4.0  # Higher contrast
MASK_CLAHE_TILE_SIZE = (8, 8)  # Smoother enhancement
```

### Mask Only (No Enhancement)
```python
USE_CENTER_SQUARE_MASK = True
USE_MASK_REGION_ENHANCEMENT = False
```

## Integration

The mask is automatically applied in:
- `eye_tracking.py` - `process_frame()` function
- `eye_tracking.py` - `run_eye_tracking()` function
- `main.py` - Main processing loop

## API

### Functions

#### `apply_center_square_mask(image, center_x=None, center_y=None, size=None)`
Apply the mask to an image.

**Args:**
- `image`: Input grayscale image
- `center_x`: X coordinate of square center (uses `CENTER_SQUARE_X` or frame center if None)
- `center_y`: Y coordinate of square center (uses `CENTER_SQUARE_Y` or frame center if None)
- `size`: Size of square in pixels (uses `CENTER_SQUARE_SIZE` if None)

**Returns:**
- Masked image with white pixels outside the square

#### `apply_mask_if_enabled(image)`
Apply mask if `USE_CENTER_SQUARE_MASK` is True.

**Args:**
- `image`: Input grayscale image

**Returns:**
- Masked image (or original if mask disabled)

#### `enhance_mask_region(image)`
Apply gamma correction and CLAHE to the square region only.

**Args:**
- `image`: Input grayscale image (already masked)

**Returns:**
- Enhanced image with only the square region enhanced (white areas unchanged)

## Notes

- The mask assumes a fixed frame size
- Boundary checking ensures the square stays within frame bounds
- White pixels (255) outside the square prevent them from being detected as dark regions

