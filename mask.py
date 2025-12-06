"""
Mask module for eye tracking video processing.

Provides functions for applying masks to video frames.
"""

import cv2
import numpy as np

# Center square mask parameters
USE_CENTER_SQUARE_MASK = True  # Enable/disable center square mask
CENTER_SQUARE_SIZE = 250  # Size of the square in pixels (200x200)
CENTER_SQUARE_X = 250  # X center position (None = frame center)
CENTER_SQUARE_Y = 300  # Y center position (None = frame center)

# Mask region enhancement parameters
USE_MASK_REGION_ENHANCEMENT = True  # Enable/disable enhancement of masked region
MASK_GAMMA_VALUE = 0.5  # Gamma value for masked region (lower = brighter dark areas)
MASK_CLAHE_CLIP_LIMIT = 5.0  # CLAHE clip limit for masked region
MASK_CLAHE_TILE_SIZE = (4, 4)  # CLAHE tile size for masked region (smaller = more adaptive)

def apply_center_square_mask(image, center_x=None, center_y=None, size=None):
    """
    Apply a mask that keeps only a square region at the center, rest becomes white.
    
    Args:
        image: Input grayscale image
        center_x: X coordinate of square center (None = uses CENTER_SQUARE_X or image center)
        center_y: Y coordinate of square center (None = uses CENTER_SQUARE_Y or image center)
        size: Size of the square in pixels (uses CENTER_SQUARE_SIZE if None)
    
    Returns:
        Masked image with white pixels outside the square
    """
    if size is None:
        size = CENTER_SQUARE_SIZE
    
    masked_image = image.copy()
    h, w = image.shape[:2]
    
    # Use configured center or image center if not specified
    if center_x is None:
        center_x = CENTER_SQUARE_X if CENTER_SQUARE_X is not None else w // 2
    if center_y is None:
        center_y = CENTER_SQUARE_Y if CENTER_SQUARE_Y is not None else h // 2
    
    # Calculate square boundaries
    half_size = size // 2
    top_left_x = max(0, center_x - half_size)
    top_left_y = max(0, center_y - half_size)
    bottom_right_x = min(w, center_x + half_size)
    bottom_right_y = min(h, center_y + half_size)
    
    # Create mask: white (255) everywhere, black (0) in square region
    mask = np.ones_like(image, dtype=np.uint8) * 255
    
    # Set square region to black (0) in mask
    mask[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = 0
    
    # Apply mask: where mask is white (255), set image to white; where mask is black (0), keep original
    masked_image = np.where(mask == 255, 255, masked_image)
    
    return masked_image

def apply_mask_if_enabled(image):
    """
    Apply center square mask if enabled via config flag.
    
    Args:
        image: Input grayscale image
    
    Returns:
        Masked image (or original if mask disabled)
    """
    if USE_CENTER_SQUARE_MASK:
        return apply_center_square_mask(image)
    return image

def enhance_mask_region(image):
    """
    Apply gamma correction and CLAHE only to the square region within the mask.
    The white pixels outside remain white.
    
    Args:
        image: Input grayscale image (already masked)
    
    Returns:
        Enhanced image with only the square region enhanced
    """
    if not USE_MASK_REGION_ENHANCEMENT:
        return image
    
    enhanced = image.copy()
    h, w = image.shape[:2]
    
    # Get mask center and size
    center_x, center_y, size = get_mask_center_and_size(image)
    half_size = size // 2
    
    # Calculate square boundaries
    top_left_x = max(0, center_x - half_size)
    top_left_y = max(0, center_y - half_size)
    bottom_right_x = min(w, center_x + half_size)
    bottom_right_y = min(h, center_y + half_size)
    
    # Extract the square region
    square_region = image[top_left_y:bottom_right_y, top_left_x:bottom_right_x].copy()
    
    # Apply gamma correction
    inv_gamma = 1.0 / MASK_GAMMA_VALUE
    table = np.array([((i / 255.0) ** inv_gamma) * 255 
                      for i in np.arange(0, 256)]).astype("uint8")
    square_region = cv2.LUT(square_region, table)
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=MASK_CLAHE_CLIP_LIMIT, tileGridSize=MASK_CLAHE_TILE_SIZE)
    square_region = clahe.apply(square_region)
    
    # Put enhanced region back into the image
    enhanced[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = square_region
    
    return enhanced

def get_mask_center_and_size(image):
    """
    Get the center position and size of the mask for visualization.
    
    Args:
        image: Input image (to get dimensions)
    
    Returns:
        Tuple of (center_x, center_y, size)
    """
    h, w = image.shape[:2]
    center_x = CENTER_SQUARE_X if CENTER_SQUARE_X is not None else w // 2
    center_y = CENTER_SQUARE_Y if CENTER_SQUARE_Y is not None else h // 2
    size = CENTER_SQUARE_SIZE
    return center_x, center_y, size

def draw_mask_visualization(frame):
    """
    Draw the mask square and center point on a frame for debugging.
    
    Args:
        frame: Input frame (BGR or grayscale)
    
    Returns:
        Frame with mask visualization drawn on it
    """
    if not USE_CENTER_SQUARE_MASK:
        return frame
    
    # Convert to BGR if grayscale
    if len(frame.shape) == 2:
        vis_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    else:
        vis_frame = frame.copy()
    
    center_x, center_y, size = get_mask_center_and_size(vis_frame)
    half_size = size // 2
    
    # Calculate square boundaries
    h, w = vis_frame.shape[:2]
    top_left_x = max(0, center_x - half_size)
    top_left_y = max(0, center_y - half_size)
    bottom_right_x = min(w, center_x + half_size)
    bottom_right_y = min(h, center_y + half_size)
    
    # Draw square outline (green)
    cv2.rectangle(vis_frame, 
                  (top_left_x, top_left_y), 
                  (bottom_right_x, bottom_right_y),
                  (0, 255, 0),  # Green color
                  2)  # Thickness
    
    # Draw center point (red circle)
    cv2.circle(vis_frame, (center_x, center_y), 5, (0, 0, 255), -1)  # Red filled circle
    
    # Draw center crosshair
    cv2.line(vis_frame, (center_x - 10, center_y), (center_x + 10, center_y), (0, 0, 255), 1)
    cv2.line(vis_frame, (center_x, center_y - 10), (center_x, center_y + 10), (0, 0, 255), 1)
    
    # Add text label
    label = f"Mask: {size}x{size} @ ({center_x}, {center_y})"
    cv2.putText(vis_frame, label, (top_left_x, top_left_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    return vis_frame

def get_mask_center_and_size(image):
    """
    Get the center position and size of the mask for visualization.
    
    Args:
        image: Input image (to get dimensions)
    
    Returns:
        Tuple of (center_x, center_y, size)
    """
    h, w = image.shape[:2]
    center_x = CENTER_SQUARE_X if CENTER_SQUARE_X is not None else w // 2
    center_y = CENTER_SQUARE_Y if CENTER_SQUARE_Y is not None else h // 2
    size = CENTER_SQUARE_SIZE
    return center_x, center_y, size

def draw_mask_visualization(frame):
    """
    Draw the mask square and center point on a frame for debugging.
    
    Args:
        frame: Input frame (BGR or grayscale)
    
    Returns:
        Frame with mask visualization drawn on it
    """
    if not USE_CENTER_SQUARE_MASK:
        return frame
    
    # Convert to BGR if grayscale
    if len(frame.shape) == 2:
        vis_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    else:
        vis_frame = frame.copy()
    
    center_x, center_y, size = get_mask_center_and_size(vis_frame)
    half_size = size // 2
    
    # Calculate square boundaries
    h, w = vis_frame.shape[:2]
    top_left_x = max(0, center_x - half_size)
    top_left_y = max(0, center_y - half_size)
    bottom_right_x = min(w, center_x + half_size)
    bottom_right_y = min(h, center_y + half_size)
    
    # Draw square outline (green)
    cv2.rectangle(vis_frame, 
                  (top_left_x, top_left_y), 
                  (bottom_right_x, bottom_right_y),
                  (0, 255, 0),  # Green color
                  2)  # Thickness
    
    # Draw center point (red circle)
    cv2.circle(vis_frame, (center_x, center_y), 5, (0, 0, 255), -1)  # Red filled circle
    
    # Draw center crosshair
    cv2.line(vis_frame, (center_x - 10, center_y), (center_x + 10, center_y), (0, 0, 255), 1)
    cv2.line(vis_frame, (center_x, center_y - 10), (center_x, center_y + 10), (0, 0, 255), 1)
    
    # Add text label
    label = f"Mask: {size}x{size} @ ({center_x}, {center_y})"
    cv2.putText(vis_frame, label, (top_left_x, top_left_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    return vis_frame

