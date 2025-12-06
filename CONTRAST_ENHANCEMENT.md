# Contrast Enhancement for Pupil Detection

The eye tracking system now includes gamma correction and CLAHE (Contrast Limited Adaptive Histogram Equalization) to improve pupil identification and tracking.

## Features

### Gamma Correction
- Brightens dark areas (like the pupil) while preserving highlights
- Uses a lookup table for fast processing
- Configurable gamma value (default: 0.6)

### CLAHE
- Adaptive histogram equalization that improves local contrast
- Prevents over-amplification of noise
- Configurable clip limit and tile size

## Configuration

Edit these parameters at the top of `eye_tracking.py` (lines 26-30):

```python
# Contrast enhancement parameters
USE_GAMMA_CORRECTION = True  # Enable/disable gamma correction
USE_CLAHE = True  # Enable/disable CLAHE
GAMMA_VALUE = 0.6  # Gamma value (0.4-0.8 brightens, 1.0 = no change)
CLAHE_CLIP_LIMIT = 2.0  # CLAHE clip limit (1.0-4.0, higher = more contrast)
CLAHE_TILE_SIZE = (8, 8)  # CLAHE tile grid size
```

## Parameter Tuning

### Gamma Value
- **0.4-0.6**: Significant brightening of dark areas (best for low light)
- **0.6-0.8**: Moderate brightening (good default)
- **0.8-1.0**: Subtle brightening
- **1.0**: No gamma correction

### CLAHE Clip Limit
- **1.0-2.0**: Conservative contrast enhancement (less noise)
- **2.0-3.0**: Moderate enhancement (good default)
- **3.0-4.0**: Aggressive enhancement (may amplify noise)

### CLAHE Tile Size
- **(4, 4)**: More adaptive, follows local variations closely
- **(8, 8)**: Balanced (good default)
- **(16, 16)**: Smoother, less adaptive

## How It Works

1. Frame is captured and converted to grayscale
2. **Gamma correction** is applied (if enabled)
3. **CLAHE** is applied (if enabled)
4. Enhanced grayscale image is used for:
   - Finding the darkest point
   - Thresholding operations
   - Pupil contour detection

## Benefits

- Better pupil detection in varying lighting conditions
- Improved contrast between pupil and iris
- More stable tracking
- Reduced false positives from reflections

## Disabling Enhancement

To disable enhancement, set in `eye_tracking.py`:

```python
USE_GAMMA_CORRECTION = False
USE_CLAHE = False
```

Or disable individually to test each method separately.

## Testing

Run the application and observe pupil detection:

```bash
python3 main.py
```

If detection is too sensitive or not sensitive enough, adjust the parameters above.

