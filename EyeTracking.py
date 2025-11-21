import cv2
import numpy as np
import random
import math
import tkinter as tk
import os
import sys
import time
import threading
from tkinter import filedialog
import matplotlib.pyplot as plt
import argparse
import csv
import json
from datetime import datetime

# Geometric parameters
DISPLAY_DISTANCE_CM = 40.0  # Distance from eye to display in cm (adjust to your setup)
DISPLAY_DISTANCE_MM = DISPLAY_DISTANCE_CM * 10.0
DISPLAY_DPI = 96.0  # Adjust for your monitor (typical: 96, high-DPI: 144-192)
CAMERA_FOV_DEGREES = 60.0  # Camera field of view in degrees (adjust if needed)
GAZE_SCALING_FACTOR = 5.0  # Scaling factor to adjust gaze sensitivity (increase if gaze doesn't reach screen edges)
FLIP_CAMERA_HORIZONTAL = True  # Flip camera horizontally to fix mirroring

# Monitor resolution (will be auto-detected)
monitor_width = None
monitor_height = None
display_width_mm = None
display_height_mm = None

# Gaze overlay globals
gaze_overlay_window = None
gaze_overlay_canvas = None
current_gaze_x = None
current_gaze_y = None
raw_gaze_x = None  # Raw gaze position before calibration offset
raw_gaze_y = None
overlay_running = False

# Calibration globals
calibration_offset_x = 0.0  # Offset in pixels
calibration_offset_y = 0.0
calibration_active = False

# Debug calibration flag
debug_calibration = False

# Accuracy testing globals
accuracy_testing_active = False
accuracy_data = []  # List to store accuracy measurements
target_position = None  # Current target position (x, y)
target_start_time = None
target_duration = 2.0  # How long to stay at each target (seconds)
target_path = []  # List of target positions to visit
target_index = 0
accuracy_test_start_time = None

# Crop the image to maintain a specific aspect ratio (width:height) before resizing. 
def crop_to_aspect_ratio(image, width=640, height=480):
    
    # Calculate current aspect ratio
    current_height, current_width = image.shape[:2]
    desired_ratio = width / height
    current_ratio = current_width / current_height

    if current_ratio > desired_ratio:
        # Current image is too wide
        new_width = int(desired_ratio * current_height)
        offset = (current_width - new_width) // 2
        cropped_img = image[:, offset:offset+new_width]
    else:
        # Current image is too tall
        new_height = int(current_width / desired_ratio)
        offset = (current_height - new_height) // 2
        cropped_img = image[offset:offset+new_height, :]

    return cv2.resize(cropped_img, (width, height))

#apply thresholding to an image
def apply_binary_threshold(image, darkestPixelValue, addedThreshold):
    # Calculate the threshold as the sum of the two input values
    threshold = darkestPixelValue + addedThreshold
    # Apply the binary threshold
    _, thresholded_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY_INV)
    
    return thresholded_image

#Finds a square area of dark pixels in the image
#@param I input image (converted to grayscale during search process)
#@return a point within the pupil region
def get_darkest_area(image):

    ignoreBounds = 20 #don't search the boundaries of the image for ignoreBounds pixels
    imageSkipSize = 10 #only check the darkness of a block for every Nth x and y pixel (sparse sampling)
    searchArea = 20 #the size of the block to search
    internalSkipSize = 5 #skip every Nth x and y pixel in the local search area (sparse sampling)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    min_sum = float('inf')
    darkest_point = None

    # Loop over the image with spacing defined by imageSkipSize, ignoring the boundaries
    for y in range(ignoreBounds, gray.shape[0] - ignoreBounds, imageSkipSize):
        for x in range(ignoreBounds, gray.shape[1] - ignoreBounds, imageSkipSize):
            # Calculate sum of pixel values in the search area, skipping pixels based on internalSkipSize
            current_sum = np.int64(0)
            num_pixels = 0
            for dy in range(0, searchArea, internalSkipSize):
                if y + dy >= gray.shape[0]:
                    break
                for dx in range(0, searchArea, internalSkipSize):
                    if x + dx >= gray.shape[1]:
                        break
                    current_sum += gray[y + dy][x + dx]
                    num_pixels += 1

            # Update the darkest point if the current block is darker
            if current_sum < min_sum and num_pixels > 0:
                min_sum = current_sum
                darkest_point = (x + searchArea // 2, y + searchArea // 2)  # Center of the block

    return darkest_point

#mask all pixels outside a square defined by center and size
def mask_outside_square(image, center, size):
    x, y = center
    half_size = size // 2

    # Create a mask initialized to black
    mask = np.zeros_like(image)

    # Calculate the top-left corner of the square
    top_left_x = max(0, x - half_size)
    top_left_y = max(0, y - half_size)

    # Calculate the bottom-right corner of the square
    bottom_right_x = min(image.shape[1], x + half_size)
    bottom_right_y = min(image.shape[0], y + half_size)

    # Set the square area in the mask to white
    mask[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = 255

    # Apply the mask to the image
    masked_image = cv2.bitwise_and(image, mask)

    return masked_image
   
def optimize_contours_by_angle(contours, image):
    if len(contours) < 1:
        return contours

    # Holds the candidate points
    all_contours = np.concatenate(contours[0], axis=0)

    # Set spacing based on size of contours
    spacing = int(len(all_contours)/25)  # Spacing between sampled points

    # Temporary array for result
    filtered_points = []
    
    # Calculate centroid of the original contours
    centroid = np.mean(all_contours, axis=0)
    
    # Create an image of the same size as the original image
    point_image = image.copy()
    
    skip = 0
    
    # Loop through each point in the all_contours array
    for i in range(0, len(all_contours), 1):
    
        # Get three points: current point, previous point, and next point
        current_point = all_contours[i]
        prev_point = all_contours[i - spacing] if i - spacing >= 0 else all_contours[-spacing]
        next_point = all_contours[i + spacing] if i + spacing < len(all_contours) else all_contours[spacing]
        
        # Calculate vectors between points
        vec1 = prev_point - current_point
        vec2 = next_point - current_point
        
        with np.errstate(invalid='ignore'):
            # Calculate angles between vectors
            angle = np.arccos(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

        
        # Calculate vector from current point to centroid
        vec_to_centroid = centroid - current_point
        
        # Check if angle is oriented towards centroid
        # Calculate the cosine of the desired angle threshold (e.g., 80 degrees)
        cos_threshold = np.cos(np.radians(60))  # Convert angle to radians
        
        if np.dot(vec_to_centroid, (vec1+vec2)/2) >= cos_threshold:
            filtered_points.append(current_point)
    
    return np.array(filtered_points, dtype=np.int32).reshape((-1, 1, 2))

#returns the largest contour that is not extremely long or tall
#contours is the list of contours, pixel_thresh is the max pixels to filter, and ratio_thresh is the max ratio
def filter_contours_by_area_and_return_largest(contours, pixel_thresh, ratio_thresh):
    max_area = 0
    largest_contour = None
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= pixel_thresh:
            x, y, w, h = cv2.boundingRect(contour)
            length = max(w, h)
            width = min(w, h)

            # Calculate the length-to-width ratio and width-to-length ratio
            length_to_width_ratio = length / width
            width_to_length_ratio = width / length

            # Pick the higher of the two ratios
            current_ratio = max(length_to_width_ratio, width_to_length_ratio)

            # Check if highest ratio is within the acceptable threshold
            if current_ratio <= ratio_thresh:
                # Update the largest contour if the current one is bigger
                if area > max_area:
                    max_area = area
                    largest_contour = contour

    # Return a list with only the largest contour, or an empty list if no contour was found
    if largest_contour is not None:
        return [largest_contour]
    else:
        return []

#Fits an ellipse to the optimized contours and draws it on the image.
def fit_and_draw_ellipses(image, optimized_contours, color):
    if len(optimized_contours) >= 5:
        # Ensure the data is in the correct shape (n, 1, 2) for cv2.fitEllipse
        contour = np.array(optimized_contours, dtype=np.int32).reshape((-1, 1, 2))

        # Fit ellipse
        ellipse = cv2.fitEllipse(contour)

        # Draw the ellipse
        cv2.ellipse(image, ellipse, color, 2)  # Draw with green color and thickness of 2

        return image
    else:
        print("Not enough points to fit an ellipse.")
        return image

#checks how many pixels in the contour fall under a slightly thickened ellipse
#also returns that number of pixels divided by the total pixels on the contour border
#assists with checking ellipse goodness    
def check_contour_pixels(contour, image_shape, debug_mode_on):
    # Check if the contour can be used to fit an ellipse (requires at least 5 points)
    if len(contour) < 5:
        return [0, 0]  # Not enough points to fit an ellipse
    
    # Create an empty mask for the contour
    contour_mask = np.zeros(image_shape, dtype=np.uint8)
    # Draw the contour on the mask, filling it
    cv2.drawContours(contour_mask, [contour], -1, (255), 1)
   
    # Fit an ellipse to the contour and create a mask for the ellipse
    ellipse_mask_thick = np.zeros(image_shape, dtype=np.uint8)
    ellipse_mask_thin = np.zeros(image_shape, dtype=np.uint8)
    ellipse = cv2.fitEllipse(contour)
    
    # Draw the ellipse with a specific thickness
    cv2.ellipse(ellipse_mask_thick, ellipse, (255), 10) #capture more for absolute
    cv2.ellipse(ellipse_mask_thin, ellipse, (255), 4) #capture fewer for ratio

    # Calculate the overlap of the contour mask and the thickened ellipse mask
    overlap_thick = cv2.bitwise_and(contour_mask, ellipse_mask_thick)
    overlap_thin = cv2.bitwise_and(contour_mask, ellipse_mask_thin)
    
    # Count the number of non-zero (white) pixels in the overlap
    absolute_pixel_total_thick = np.sum(overlap_thick > 0)#compute with thicker border
    absolute_pixel_total_thin = np.sum(overlap_thin > 0)#compute with thicker border
    
    # Compute the ratio of pixels under the ellipse to the total pixels on the contour border
    total_border_pixels = np.sum(contour_mask > 0)
    
    ratio_under_ellipse = absolute_pixel_total_thin / total_border_pixels if total_border_pixels > 0 else 0
    
    return [absolute_pixel_total_thick, ratio_under_ellipse, overlap_thin]

#outside of this method, select the ellipse with the highest percentage of pixels under the ellipse 
#TODO for efficiency, work with downscaled or cropped images
def check_ellipse_goodness(binary_image, contour, debug_mode_on):
    ellipse_goodness = [0,0,0] #covered pixels, edge straightness stdev, skewedness   
    # Check if the contour can be used to fit an ellipse (requires at least 5 points)
    if len(contour) < 5:
        print("length of contour was 0")
        return 0  # Not enough points to fit an ellipse
    
    # Fit an ellipse to the contour
    ellipse = cv2.fitEllipse(contour)
    
    # Create a mask with the same dimensions as the binary image, initialized to zero (black)
    mask = np.zeros_like(binary_image)
    
    # Draw the ellipse on the mask with white color (255)
    cv2.ellipse(mask, ellipse, (255), -1)
    
    # Calculate the number of pixels within the ellipse
    ellipse_area = np.sum(mask == 255)
    
    # Calculate the number of white pixels within the ellipse
    covered_pixels = np.sum((binary_image == 255) & (mask == 255))
    
    # Calculate the percentage of covered white pixels within the ellipse
    if ellipse_area == 0:
        print("area was 0")
        return ellipse_goodness  # Avoid division by zero if the ellipse area is somehow zero
    
    #percentage of covered pixels to number of pixels under area
    ellipse_goodness[0] = covered_pixels / ellipse_area
    
    #skew of the ellipse (less skewed is better?) - may not need this
    axes_lengths = ellipse[1]  # This is a tuple (minor_axis_length, major_axis_length)
    major_axis_length = axes_lengths[1]
    minor_axis_length = axes_lengths[0]
    ellipse_goodness[2] = min(ellipse[1][1]/ellipse[1][0], ellipse[1][0]/ellipse[1][1])
    
    return ellipse_goodness

# ------------------- Gaze Tracking Functions -------------------
def get_monitor_resolution():
    """Get primary monitor resolution"""
    global monitor_width, monitor_height
    try:
        root = tk.Tk()
        monitor_width = root.winfo_screenwidth()
        monitor_height = root.winfo_screenheight()
        root.destroy()
        return monitor_width, monitor_height
    except:
        print("using default monitor resolution")
        monitor_width, monitor_height = 1920, 1080  # Default fallback
        return monitor_width, monitor_height

def calculate_display_dimensions():
    """Calculate display dimensions in mm based on resolution and DPI"""
    global monitor_width, monitor_height, display_width_mm, display_height_mm, DISPLAY_DPI
    if monitor_width is None:
        get_monitor_resolution()
    
    pixels_per_mm = DISPLAY_DPI / 25.4
    display_width_mm = monitor_width / pixels_per_mm
    display_height_mm = monitor_height / pixels_per_mm
    print(f"Display: {display_width_mm:.1f}mm × {display_height_mm:.1f}mm at {DISPLAY_DISTANCE_CM}cm distance")

def compute_gaze_to_screen(pupil_x, pupil_y, frame_width, frame_height, apply_calibration=True):
    """
    Convert pupil position to screen coordinates using geometric projection.
    
    Assumes:
    - Eye center is at frame center
    - Camera FOV is set by CAMERA_FOV_DEGREES
    - Display is at DISPLAY_DISTANCE_MM from eye
    
    Returns:
    - (screen_x, screen_y): Screen coordinates with calibration applied if apply_calibration=True
    """
    global display_width_mm, display_height_mm, monitor_width, monitor_height, DISPLAY_DISTANCE_MM
    global calibration_offset_x, calibration_offset_y, CAMERA_FOV_DEGREES
    
    if display_width_mm is None:
        calculate_display_dimensions()
    
    # Frame center (assumed eye center)
    eye_center_x = frame_width / 2.0
    eye_center_y = frame_height / 2.0
    
    # Offset from eye center to pupil (in pixels)
    offset_x = pupil_x - eye_center_x
    offset_y = pupil_y - eye_center_y
    
    # Convert pixel offset to angle using actual FOV
    FOV_rad = np.radians(CAMERA_FOV_DEGREES)
    
    # Normalized device coordinates [-1, 1]
    # Map from frame coordinates to normalized coordinates
    ndc_x = (2.0 * pupil_x / frame_width) - 1.0
    ndc_y = 1.0 - (2.0 * pupil_y / frame_height)  # Invert Y axis
    
    # Convert to angle (approximate, using FOV)
    angle_x = ndc_x * (FOV_rad / 2.0)
    angle_y = ndc_y * (FOV_rad / 2.0)
    
    # Project onto display plane at DISPLAY_DISTANCE_MM
    # tan(angle) = offset / distance
    # Apply scaling factor to adjust sensitivity
    global GAZE_SCALING_FACTOR
    screen_x_mm = np.tan(angle_x) * DISPLAY_DISTANCE_MM * GAZE_SCALING_FACTOR
    screen_y_mm = np.tan(angle_y) * DISPLAY_DISTANCE_MM * GAZE_SCALING_FACTOR
    
    # Convert to screen coordinates (center of screen is origin)
    screen_x_mm += display_width_mm / 2.0
    screen_y_mm = (display_height_mm / 2.0) - screen_y_mm  # Invert Y for screen coords
    
    # Convert mm to pixels
    pixels_per_mm_x = monitor_width / display_width_mm
    pixels_per_mm_y = monitor_height / display_height_mm
    
    # Raw gaze position (before calibration)
    raw_screen_x = int(screen_x_mm * pixels_per_mm_x)
    raw_screen_y = int(screen_y_mm * pixels_per_mm_y)
    
    # Apply calibration offset if requested
    if apply_calibration:
        screen_x = raw_screen_x + int(calibration_offset_x)
        screen_y = raw_screen_y + int(calibration_offset_y)
        
        # Clamp to screen bounds
        screen_x = max(0, min(monitor_width - 1, screen_x))
        screen_y = max(0, min(monitor_height - 1, screen_y))
    else:
        screen_x = raw_screen_x
        screen_y = raw_screen_y
    
    return screen_x, screen_y

def calibrate_gaze(raw_screen_x, raw_screen_y):
    """Calibrate gaze by computing offset from raw gaze position to screen center"""
    global calibration_offset_x, calibration_offset_y, monitor_width, monitor_height
    
    if monitor_width is None:
        get_monitor_resolution()
    
    # Compute offset needed to center the gaze
    # Use raw gaze position (before any calibration offset)
    screen_center_x = monitor_width // 2
    screen_center_y = monitor_height // 2
    
    # Calculate offset: what we need to add to raw gaze to get to center
    calibration_offset_x = screen_center_x - raw_screen_x
    calibration_offset_y = screen_center_y - raw_screen_y
    
    print(f"Calibration complete!")
    print(f"  Raw gaze was at: ({raw_screen_x}, {raw_screen_y})")
    print(f"  Screen center is: ({screen_center_x}, {screen_center_y})")
    print(f"  Applied offset: ({calibration_offset_x:.1f}, {calibration_offset_y:.1f})")
    print("  After calibration, gaze should be at screen center when looking at center.")

# ------------------- Gaze Overlay Functions -------------------
def create_gaze_overlay():
    """Create transparent overlay window for gaze dot"""
    global gaze_overlay_window, gaze_overlay_canvas, monitor_width, monitor_height
    
    if monitor_width is None:
        get_monitor_resolution()
    
    # Close existing window if any
    if gaze_overlay_window is not None:
        try:
            gaze_overlay_window.destroy()
        except:
            pass
    
    root = tk.Tk()
    root.title("Gaze Overlay")
    root.attributes('-fullscreen', True)
    root.attributes('-topmost', True)
    root.configure(bg='black')
    root.attributes('-alpha', 0.3)  # Semi-transparent
    root.overrideredirect(True)  # Remove window decorations
    
    # Try to make it stay on top
    try:
        root.lift()
        root.attributes('-topmost', True)
        if sys.platform == "darwin":  # macOS
            root.call('wm', 'attributes', '.', '-topmost', True)
    except:
        pass
    
    canvas = tk.Canvas(root, width=monitor_width, height=monitor_height, 
                      bg='black', highlightthickness=0)
    canvas.pack(fill=tk.BOTH, expand=True)
    
    gaze_overlay_window = root
    gaze_overlay_canvas = canvas
    
    return root, canvas

def update_gaze_overlay(screen_x, screen_y):
    """Update red dot position on overlay"""
    global gaze_overlay_window, gaze_overlay_canvas, debug_calibration, monitor_width, monitor_height
    global accuracy_testing_active, target_position, target_index, target_path
    
    if gaze_overlay_window is None or gaze_overlay_canvas is None:
        return
    
    if screen_x is None or screen_y is None:
        return
    
    try:
        # Check if window still exists
        if not hasattr(gaze_overlay_window, 'winfo_exists') or not gaze_overlay_window.winfo_exists():
            return
        
        # Clear and redraw
        gaze_overlay_canvas.delete("all")
        
        # Draw accuracy testing target if active
        if accuracy_testing_active and target_position is not None:
            target_x, target_y = target_position
            # Draw large target circle (green)
            target_radius = 30
            gaze_overlay_canvas.create_oval(
                target_x - target_radius, target_y - target_radius,
                target_x + target_radius, target_y + target_radius,
                fill='green', outline='darkgreen', width=3
            )
            # Draw inner white dot
            gaze_overlay_canvas.create_oval(
                target_x - 8, target_y - 8,
                target_x + 8, target_y + 8,
                fill='white', outline='white'
            )
            # Show target number
            if target_path:
                gaze_overlay_canvas.create_text(
                    target_x, target_y - target_radius - 20,
                    text=f"Target {target_index + 1}/{len(target_path)}",
                    fill='green', font=('Arial', 14, 'bold')
                )
        
        # Draw red dot (gaze position)
        dot_radius = 20
        gaze_overlay_canvas.create_oval(
            screen_x - dot_radius, screen_y - dot_radius,
            screen_x + dot_radius, screen_y + dot_radius,
            fill='red', outline='darkred', width=3
        )
        
        # Inner white dot for precision
        gaze_overlay_canvas.create_oval(
            screen_x - 4, screen_y - 4,
            screen_x + 4, screen_y + 4,
            fill='white', outline='white'
        )
        
        # Draw line from target to gaze if accuracy testing
        if accuracy_testing_active and target_position is not None:
            target_x, target_y = target_position
            gaze_overlay_canvas.create_line(
                target_x, target_y, screen_x, screen_y,
                fill='yellow', width=2, dash=(5, 5)
            )
        
        # Debug calibration: draw ovals at key points
        if debug_calibration:
            # Draw oval at gaze position (cyan)
            gaze_radius = 30
            gaze_overlay_canvas.create_oval(
                screen_x - gaze_radius, screen_y - gaze_radius,
                screen_x + gaze_radius, screen_y + gaze_radius,
                outline='cyan', width=3
            )
            
            # Draw oval at screen center (green)
            if monitor_width and monitor_height:
                screen_center_x = monitor_width // 2
                screen_center_y = monitor_height // 2
                center_radius = 40
                gaze_overlay_canvas.create_oval(
                    screen_center_x - center_radius, screen_center_y - center_radius,
                    screen_center_x + center_radius, screen_center_y + center_radius,
                    outline='green', width=4
                )
                # Label for screen center
                gaze_overlay_canvas.create_text(
                    screen_center_x, screen_center_y - center_radius - 20,
                    text="Screen Center", fill='green', font=('Arial', 12, 'bold')
                )
                # Label for gaze position
                gaze_overlay_canvas.create_text(
                    screen_x, screen_y - gaze_radius - 20,
                    text="Gaze", fill='cyan', font=('Arial', 12, 'bold')
                )
        
        # Force canvas update
        gaze_overlay_canvas.update_idletasks()
        
    except (tk.TclError, AttributeError):
        # Window was destroyed - silently handle
        pass
    except Exception:
        # Other errors - silently handle
        pass

def start_gaze_overlay():
    """Start overlay - create window on main thread, update in background"""
    global overlay_running, gaze_overlay_window
    
    if overlay_running and gaze_overlay_window is not None:
        print("Overlay already running!")
        return
    
    # Create window on main thread immediately (required for macOS)
    try:
        root, canvas = create_gaze_overlay()
        
        # Draw initial test dot at center
        if monitor_width and monitor_height:
            test_x, test_y = monitor_width // 2, monitor_height // 2
            canvas.create_oval(test_x - 30, test_y - 30, test_x + 30, test_y + 30,
                             fill='red', outline='darkred', width=3)
            root.update_idletasks()
            root.update()
            print(f"Test dot drawn at center ({test_x}, {test_y})")
    except Exception as e:
        print(f"Error creating overlay window: {e}")
        import traceback
        traceback.print_exc()
        return
    
    overlay_running = True
    
    # Periodic update using after() callback (runs on main thread)
    update_count = [0]  # Track updates for debugging
    last_gaze_x = [None]  # Track last position to detect changes
    last_gaze_y = [None]
    
    def periodic_update():
        global current_gaze_x, current_gaze_y, overlay_running, gaze_overlay_window
        if overlay_running and gaze_overlay_window is not None:
            try:
                if current_gaze_x is not None and current_gaze_y is not None:
                    # Always update, even if position hasn't changed (to ensure dot is drawn)
                    update_gaze_overlay(current_gaze_x, current_gaze_y)
                    
                    # Debug: print when position changes or occasionally
                    update_count[0] += 1
                    position_changed = (last_gaze_x[0] != current_gaze_x or last_gaze_y[0] != current_gaze_y)
                    last_gaze_x[0] = current_gaze_x
                    last_gaze_y[0] = current_gaze_y
                    
                    if position_changed or update_count[0] % 50 == 0:  # Print when changed or every 50 updates
                        print(f"Overlay update #{update_count[0]}: Gaze at ({current_gaze_x}, {current_gaze_y})")
                else:
                    # Debug: print when no gaze data
                    update_count[0] += 1
                    if update_count[0] % 100 == 0:
                        print("Overlay waiting for gaze data... (current_gaze_x is None)")
                
                # Keep window on top
                try:
                    gaze_overlay_window.lift()
                    gaze_overlay_window.attributes('-topmost', True)
                except:
                    pass
                
                # Schedule next update (20ms = ~50Hz)
                gaze_overlay_window.after(20, periodic_update)
            except (tk.TclError, AttributeError) as e:
                print(f"Overlay periodic update error: {e}")
                overlay_running = False
            except Exception as e:
                print(f"Unexpected overlay update error: {e}")
                import traceback
                traceback.print_exc()
                # Continue on other errors
                if overlay_running and gaze_overlay_window is not None:
                    try:
                        gaze_overlay_window.after(20, periodic_update)
                    except:
                        pass
    
    # Start periodic updates
    root.after(0, periodic_update)
    
    # Process tkinter events in background thread
    def tkinter_event_loop():
        global gaze_overlay_window, overlay_running
        while overlay_running and gaze_overlay_window is not None:
            try:
                gaze_overlay_window.update_idletasks()
                gaze_overlay_window.update()
                time.sleep(0.01)
            except (tk.TclError, AttributeError):
                break
            except Exception:
                pass
    
    thread = threading.Thread(target=tkinter_event_loop, daemon=True)
    thread.start()
    
    print("Gaze overlay started! Red dot shows where you're looking.")
    print("Press 'c' to calibrate (look at screen center and press 'c')")
    print("Press 'g' to toggle overlay on/off")
    print("Press 'a' to start/stop accuracy testing")

def stop_gaze_overlay():
    """Stop and close overlay"""
    global overlay_running, gaze_overlay_window, gaze_overlay_canvas
    
    overlay_running = False
    
    if gaze_overlay_window is not None:
        try:
            gaze_overlay_window.destroy()
        except:
            pass
        gaze_overlay_window = None
        gaze_overlay_canvas = None
    
    print("Gaze overlay stopped.")

# ------------------- Accuracy Testing Functions -------------------
def generate_target_path(grid_size=3, margin_percent=0.15):
    """
    Generate a grid of target positions across the screen.
    
    Args:
        grid_size: Number of targets per row/column (default 3x3 = 9 targets)
        margin_percent: Percentage of screen to leave as margin (default 15%)
    
    Returns:
        List of (x, y) tuples representing target positions
    """
    global monitor_width, monitor_height
    
    if monitor_width is None or monitor_height is None:
        get_monitor_resolution()
    
    margin_x = int(monitor_width * margin_percent)
    margin_y = int(monitor_height * margin_percent)
    
    usable_width = monitor_width - 2 * margin_x
    usable_height = monitor_height - 2 * margin_y
    
    targets = []
    for row in range(grid_size):
        for col in range(grid_size):
            x = margin_x + int((col / (grid_size - 1)) * usable_width) if grid_size > 1 else margin_x + usable_width // 2
            y = margin_y + int((row / (grid_size - 1)) * usable_height) if grid_size > 1 else margin_y + usable_height // 2
            targets.append((x, y))
    
    return targets

def start_accuracy_testing(grid_size=3):
    """Start accuracy testing with moving target"""
    global accuracy_testing_active, target_path, target_index, target_position
    global target_start_time, accuracy_data, accuracy_test_start_time, monitor_width, monitor_height
    
    if monitor_width is None or monitor_height is None:
        get_monitor_resolution()
    
    # Generate target path
    target_path = generate_target_path(grid_size)
    target_index = 0
    target_position = target_path[0]
    target_start_time = time.time()
    accuracy_test_start_time = time.time()
    accuracy_data = []
    accuracy_testing_active = True
    
    print(f"Accuracy testing started!")
    print(f"  Grid size: {grid_size}x{grid_size} ({len(target_path)} targets)")
    print(f"  Duration per target: {target_duration:.1f} seconds")
    print(f"  Total estimated time: {len(target_path) * target_duration:.1f} seconds")
    print(f"  Press 'a' again to stop early")

def stop_accuracy_testing():
    """Stop accuracy testing and calculate results"""
    global accuracy_testing_active, accuracy_data, target_path
    
    if not accuracy_testing_active:
        return
    
    accuracy_testing_active = False
    target_position = None
    
    if len(accuracy_data) == 0:
        print("No accuracy data collected.")
        return
    
    # Calculate accuracy metrics
    results = calculate_accuracy_metrics(accuracy_data)
    
    # Save data
    save_accuracy_data(accuracy_data, results)
    
    # Print results
    print_accuracy_results(results)
    
    return results

def update_accuracy_target():
    """Update target position based on time"""
    global target_position, target_index, target_path, target_start_time, target_duration
    
    if not accuracy_testing_active or len(target_path) == 0:
        return
    
    current_time = time.time()
    elapsed = current_time - target_start_time
    
    # Move to next target if duration exceeded
    if elapsed >= target_duration:
        target_index += 1
        if target_index >= len(target_path):
            # All targets visited, stop testing
            stop_accuracy_testing()
            return
        target_position = target_path[target_index]
        target_start_time = current_time

def record_accuracy_measurement(gaze_x, gaze_y):
    """Record a single accuracy measurement"""
    global accuracy_data, target_position, accuracy_testing_active
    
    if not accuracy_testing_active or target_position is None:
        return
    
    if gaze_x is None or gaze_y is None:
        return
    
    # Calculate error
    target_x, target_y = target_position
    error_x = gaze_x - target_x
    error_y = gaze_y - target_y
    error_distance = math.sqrt(error_x**2 + error_y**2)
    
    # Record measurement
    measurement = {
        'timestamp': time.time(),
        'target_x': target_x,
        'target_y': target_y,
        'gaze_x': gaze_x,
        'gaze_y': gaze_y,
        'error_x': error_x,
        'error_y': error_y,
        'error_distance': error_distance,
        'target_index': target_index
    }
    
    accuracy_data.append(measurement)

def calculate_accuracy_metrics(data):
    """Calculate accuracy metrics from collected data"""
    if len(data) == 0:
        return None
    
    errors = [d['error_distance'] for d in data]
    errors_x = [d['error_x'] for d in data]
    errors_y = [d['error_y'] for d in data]
    
    mean_error = np.mean(errors)
    median_error = np.median(errors)
    std_error = np.std(errors)
    max_error = np.max(errors)
    min_error = np.min(errors)
    
    # Calculate percentage within different thresholds (in pixels)
    thresholds = [50, 100, 150, 200]
    within_threshold = {}
    for threshold in thresholds:
        count = sum(1 for e in errors if e <= threshold)
        within_threshold[f'within_{threshold}px'] = (count / len(errors)) * 100
    
    # Calculate RMS (Root Mean Square) error
    rms_error = np.sqrt(np.mean([e**2 for e in errors]))
    
    # Calculate mean absolute error for X and Y
    mean_error_x = np.mean([abs(e) for e in errors_x])
    mean_error_y = np.mean([abs(e) for e in errors_y])
    
    # Calculate accuracy score (0-100, higher is better)
    # Score based on how close to perfect (0 error)
    # Using inverse of mean error, normalized
    max_reasonable_error = 500  # pixels - adjust based on screen size
    accuracy_score = max(0, min(100, 100 * (1 - mean_error / max_reasonable_error)))
    
    results = {
        'total_measurements': len(data),
        'mean_error_pixels': mean_error,
        'median_error_pixels': median_error,
        'std_error_pixels': std_error,
        'max_error_pixels': max_error,
        'min_error_pixels': min_error,
        'rms_error_pixels': rms_error,
        'mean_error_x_pixels': mean_error_x,
        'mean_error_y_pixels': mean_error_y,
        'within_threshold_percent': within_threshold,
        'accuracy_score': accuracy_score,
        'test_duration_seconds': data[-1]['timestamp'] - data[0]['timestamp'] if len(data) > 1 else 0
    }
    
    return results

def save_accuracy_data(data, results):
    """Save accuracy data to CSV and JSON files"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save to CSV
    csv_filename = f"accuracy_data_{timestamp}.csv"
    try:
        with open(csv_filename, 'w', newline='') as csvfile:
            fieldnames = ['timestamp', 'target_x', 'target_y', 'gaze_x', 'gaze_y', 
                         'error_x', 'error_y', 'error_distance', 'target_index']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        print(f"✓ Accuracy data saved to {csv_filename}")
    except Exception as e:
        print(f"✗ Error saving CSV: {e}")
    
    # Save to JSON (include both raw data and results)
    json_filename = f"accuracy_results_{timestamp}.json"
    try:
        json_data = {
            'metadata': {
                'timestamp': timestamp,
                'test_date': datetime.now().isoformat(),
                'monitor_width': monitor_width,
                'monitor_height': monitor_height,
                'target_count': len(target_path) if target_path else 0
            },
            'results': results,
            'raw_data': data
        }
        with open(json_filename, 'w') as jsonfile:
            json.dump(json_data, jsonfile, indent=2)
        print(f"✓ Accuracy results saved to {json_filename}")
    except Exception as e:
        print(f"✗ Error saving JSON: {e}")

def print_accuracy_results(results):
    """Print accuracy results to console"""
    if results is None:
        return
    
    print("\n" + "="*60)
    print("ACCURACY TEST RESULTS")
    print("="*60)
    print(f"Total Measurements: {results['total_measurements']}")
    print(f"Test Duration: {results['test_duration_seconds']:.2f} seconds")
    print(f"\nError Statistics (pixels):")
    print(f"  Mean Error:       {results['mean_error_pixels']:.2f} px")
    print(f"  Median Error:      {results['median_error_pixels']:.2f} px")
    print(f"  RMS Error:         {results['rms_error_pixels']:.2f} px")
    print(f"  Std Deviation:    {results['std_error_pixels']:.2f} px")
    print(f"  Max Error:         {results['max_error_pixels']:.2f} px")
    print(f"  Min Error:         {results['min_error_pixels']:.2f} px")
    print(f"\nDirectional Errors:")
    print(f"  Mean X Error:      {results['mean_error_x_pixels']:.2f} px")
    print(f"  Mean Y Error:      {results['mean_error_y_pixels']:.2f} px")
    print(f"\nAccuracy Within Thresholds:")
    for threshold, percent in results['within_threshold_percent'].items():
        threshold_px = threshold.replace('within_', '').replace('px', '')
        print(f"  Within {threshold_px}px: {percent:.1f}%")
    print(f"\n{'='*60}")
    print(f"ACCURACY SCORE: {results['accuracy_score']:.1f}/100")
    print(f"{'='*60}\n")

def process_frames(thresholded_image_strict, thresholded_image_medium, thresholded_image_relaxed, frame, gray_frame, darkest_point, debug_mode_on, render_cv_window, debug_calibration_flag=False):
  
    final_rotated_rect = ((0,0),(0,0),0)

    image_array = [thresholded_image_relaxed, thresholded_image_medium, thresholded_image_strict] #holds images
    name_array = ["relaxed", "medium", "strict"] #for naming windows
    final_image = image_array[0] #holds return array
    final_contours = [] #holds final contours
    ellipse_reduced_contours = None  # Initialize to None instead of empty list
    goodness = 0 #goodness value for best ellipse
    best_array = 0 
    kernel_size = 5  # Size of the kernel (5x5)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    gray_copy1 = gray_frame.copy()
    gray_copy2 = gray_frame.copy()
    gray_copy3 = gray_frame.copy()
    gray_copies = [gray_copy1, gray_copy2, gray_copy3]
    final_goodness = 0
    
    #iterate through binary images and see which fits the ellipse best
    for i in range(1,4):
        # Dilate the binary image
        dilated_image = cv2.dilate(image_array[i-1], kernel, iterations=2)#medium
        
        # Find contours
        contours, hierarchy = cv2.findContours(dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create an empty image to draw contours
        contour_img2 = np.zeros_like(dilated_image)
        reduced_contours = filter_contours_by_area_and_return_largest(contours, 1000, 3)

        if len(reduced_contours) > 0 and len(reduced_contours[0]) > 5:
            current_goodness = check_ellipse_goodness(dilated_image, reduced_contours[0], debug_mode_on)
            #gray_copy = gray_frame.copy()
            #cv2.drawContours(gray_copies[i-1], reduced_contours, -1, (255), 1)
            ellipse = cv2.fitEllipse(reduced_contours[0])
            if debug_mode_on: #show contours 
                cv2.imshow(name_array[i-1] + " threshold", gray_copies[i-1])
                
            #in total pixels, first element is pixel total, next is ratio
            total_pixels = check_contour_pixels(reduced_contours[0], dilated_image.shape, debug_mode_on)                 
            
            cv2.ellipse(gray_copies[i-1], ellipse, (255, 0, 0), 2)  # Draw with specified color and thickness of 2
            font = cv2.FONT_HERSHEY_SIMPLEX  # Font type
            
            final_goodness = current_goodness[0]*total_pixels[0]*total_pixels[0]*total_pixels[1]
            
            #show intermediary images with text output
            if debug_mode_on:
                cv2.putText(gray_copies[i-1], "%filled:     " + str(current_goodness[0])[:5] + " (percentage of filled contour pixels inside ellipse)", (10,30), font, .55, (255,255,255), 1) #%filled
                cv2.putText(gray_copies[i-1], "abs. pix:   " + str(total_pixels[0]) + " (total pixels under fit ellipse)", (10,50), font, .55, (255,255,255), 1    ) #abs pix
                cv2.putText(gray_copies[i-1], "pix ratio:  " + str(total_pixels[1]) + " (total pix under fit ellipse / contour border pix)", (10,70), font, .55, (255,255,255), 1    ) #abs pix
                cv2.putText(gray_copies[i-1], "final:     " + str(final_goodness) + " (filled*ratio)", (10,90), font, .55, (255,255,255), 1) #skewedness
                cv2.imshow(name_array[i-1] + " threshold", image_array[i-1])
                cv2.imshow(name_array[i-1], gray_copies[i-1])

        if final_goodness > 0 and final_goodness > goodness: 
            goodness = final_goodness
            ellipse_reduced_contours = total_pixels[2]
            best_image = image_array[i-1]
            final_contours = reduced_contours
            final_image = dilated_image
    
    if debug_mode_on and ellipse_reduced_contours is not None:
        cv2.imshow("Reduced contours of best thresholded image", ellipse_reduced_contours)

    test_frame = frame.copy()
    
    # Only optimize if we have valid contours
    if len(final_contours) > 0:
        final_contours = [optimize_contours_by_angle(final_contours, gray_frame)]
    else:
        final_contours = []
    
    # Compute and display gaze position (always try, even if pupil detection is weak)
    global current_gaze_x, current_gaze_y
    center_x, center_y = None, None
    pupil_center = None
    
    if final_contours and len(final_contours) > 0 and not isinstance(final_contours[0], list) and len(final_contours[0]) > 5:
        #cv2.drawContours(test_frame, final_contours, -1, (255, 255, 255), 1)
        ellipse = cv2.fitEllipse(final_contours[0])
        final_rotated_rect = ellipse
        cv2.ellipse(test_frame, ellipse, (55, 255, 0), 2)
        #cv2.circle(test_frame, darkest_point, 3, (255, 125, 125), -1)
        center_x, center_y = map(int, ellipse[0])
        cv2.circle(test_frame, (center_x, center_y), 3, (255, 255, 0), -1)
        pupil_center = (center_x, center_y)
        
        # Compute gaze position
        frame_height, frame_width = frame.shape[:2]
        
        # Get raw gaze position (before calibration)
        raw_screen_x, raw_screen_y = compute_gaze_to_screen(center_x, center_y, frame_width, frame_height, apply_calibration=False)
        
        # Get calibrated gaze position (with offset applied)
        screen_x, screen_y = compute_gaze_to_screen(center_x, center_y, frame_width, frame_height, apply_calibration=True)
        
        # Store both raw and calibrated gaze positions
        global raw_gaze_x, raw_gaze_y
        raw_gaze_x = raw_screen_x
        raw_gaze_y = raw_screen_y
        current_gaze_x = screen_x
        current_gaze_y = screen_y
        
        # Debug: print occasionally to verify values are being set
        import random
        if random.random() < 0.05:  # Print 5% of the time
            frame_center_x = frame_width // 2
            frame_center_y = frame_height // 2
            offset_x = center_x - frame_center_x
            offset_y = center_y - frame_center_y
            print(f"DEBUG: Pupil at ({center_x}, {center_y}), offset: ({offset_x:.1f}, {offset_y:.1f}), "
                  f"→ Gaze: ({screen_x}, {screen_y}), screen center: ({monitor_width//2}, {monitor_height//2})")
        
        # Display gaze position on frame
        frame_center_x = frame_width // 2
        frame_center_y = frame_height // 2
        offset_from_center_x = center_x - frame_center_x
        offset_from_center_y = center_y - frame_center_y
        
        cv2.putText(test_frame, f"Gaze: ({screen_x}, {screen_y})", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        cv2.putText(test_frame, f"Pupil: ({center_x}, {center_y})", (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.putText(test_frame, f"Offset: ({offset_from_center_x}, {offset_from_center_y})", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        if calibration_offset_x != 0 or calibration_offset_y != 0:
            cv2.putText(test_frame, f"Calib: ({int(calibration_offset_x)}, {int(calibration_offset_y)})", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 0), 1)
        
        # Debug calibration: draw ovals on frame at key points
        if debug_calibration_flag:
            # Draw oval at pupil position (magenta)
            pupil_radius = 25
            cv2.circle(test_frame, (center_x, center_y), pupil_radius, (255, 0, 255), 3)
            cv2.putText(test_frame, "Pupil", (center_x - 20, center_y - pupil_radius - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
            
            # Draw oval at frame center (yellow)
            cv2.circle(test_frame, (frame_center_x, frame_center_y), 30, (0, 255, 255), 3)
            cv2.putText(test_frame, "Frame Center", (frame_center_x - 50, frame_center_y - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            # Draw line from frame center to pupil
            cv2.line(test_frame, (frame_center_x, frame_center_y), (center_x, center_y), (128, 128, 128), 2)
        
        # Debug: print occasionally to console
        import random
        if random.random() < 0.01:  # Print 1% of the time
            print(f"Frame: Pupil at ({center_x}, {center_y}), Gaze at ({screen_x}, {screen_y}), "
                  f"Offset from center: ({offset_from_center_x}, {offset_from_center_y})")
    else:
        # No pupil detected - set gaze to None or keep last known position
        # Uncomment next lines to reset gaze when pupil not detected:
        # current_gaze_x = None
        # current_gaze_y = None
        cv2.putText(test_frame, "Pupil not detected", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        if current_gaze_x is None:
            cv2.putText(test_frame, "Gaze: (None, None)", (10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    
    cv2.putText(test_frame, "SPACE = play/pause", (10,410), cv2.FONT_HERSHEY_SIMPLEX, .55, (255,90,30), 2) #space
    cv2.putText(test_frame, "Q      = quit", (10,430), cv2.FONT_HERSHEY_SIMPLEX, .55, (255,90,30), 2) #quit
    cv2.putText(test_frame, "D      = show debug", (10,450), cv2.FONT_HERSHEY_SIMPLEX, .55, (255,90,30), 2) #debug
    cv2.putText(test_frame, "C      = calibrate", (10,470), cv2.FONT_HERSHEY_SIMPLEX, .55, (255,90,30), 2) #calibrate
    cv2.putText(test_frame, "G      = toggle overlay", (200,470), cv2.FONT_HERSHEY_SIMPLEX, .55, (255,90,30), 2) #toggle overlay
    global accuracy_testing_active
    if accuracy_testing_active:
        cv2.putText(test_frame, "A      = stop accuracy test", (10,490), cv2.FONT_HERSHEY_SIMPLEX, .55, (0,255,0), 2) #accuracy test
    else:
        cv2.putText(test_frame, "A      = start accuracy test", (10,490), cv2.FONT_HERSHEY_SIMPLEX, .55, (255,90,30), 2) #accuracy test

    if render_cv_window:
        cv2.imshow('best_thresholded_image_contours_on_frame', test_frame)
    
    # Create an empty image to draw contours
    contour_img3 = np.zeros_like(image_array[i-1])
    
    # Only process final contours if they exist
    if len(final_contours) > 0 and len(final_contours[0]) >= 5:
        contour = np.array(final_contours[0], dtype=np.int32).reshape((-1, 1, 2)) #format for cv2.fitEllipse
        ellipse = cv2.fitEllipse(contour) # Fit ellipse
        cv2.ellipse(gray_frame, ellipse, (255,255,255), 2)  # Draw with white color and thickness of 2

    #process_frames now returns a rotated rectangle for the ellipse for easy access
    # Also return pupil center coordinates if available
    # pupil_center should already be set above if contours were found
    # Only set it here if it wasn't set yet
    if pupil_center is None:
        if final_rotated_rect is not None and len(final_rotated_rect) >= 3:
            try:
                pupil_center = (int(final_rotated_rect[0][0]), int(final_rotated_rect[0][1]))
            except:
                pass
    
    return final_rotated_rect, pupil_center


# Finds the pupil in an individual frame and returns the center point
def process_frame(frame):

    # Crop and resize frame
    frame = crop_to_aspect_ratio(frame)

    #find the darkest point
    darkest_point = get_darkest_area(frame)

    # Convert to grayscale to handle pixel value operations
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    darkest_pixel_value = gray_frame[darkest_point[1], darkest_point[0]]
    
    # apply thresholding operations at different levels
    # at least one should give us a good ellipse segment
    thresholded_image_strict = apply_binary_threshold(gray_frame, darkest_pixel_value, 5)#lite
    thresholded_image_strict = mask_outside_square(thresholded_image_strict, darkest_point, 250)

    thresholded_image_medium = apply_binary_threshold(gray_frame, darkest_pixel_value, 15)#medium
    thresholded_image_medium = mask_outside_square(thresholded_image_medium, darkest_point, 250)
    
    thresholded_image_relaxed = apply_binary_threshold(gray_frame, darkest_pixel_value, 25)#heavy
    thresholded_image_relaxed = mask_outside_square(thresholded_image_relaxed, darkest_point, 250)
    
    #take the three images thresholded at different levels and process them
    final_rotated_rect, pupil_center = process_frames(thresholded_image_strict, thresholded_image_medium, thresholded_image_relaxed, frame, gray_frame, darkest_point, False, False)
    
    return final_rotated_rect, pupil_center

# Loads a video and finds the pupil in each frame
def process_video(video_path, input_method, debug_calibration_flag=False):

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
    out = cv2.VideoWriter('output_video.mp4', fourcc, 30.0, (640, 480))  # Output video filename, codec, frame rate, and frame size

    if input_method == 1:
        cap = cv2.VideoCapture(video_path)
    elif input_method == 2:
        cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)  # Camera input UPDATED FOR MAC
        cap.set(cv2.CAP_PROP_EXPOSURE, -5)
    else:
        print("Invalid video source.")
        return

    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    # Initialize display dimensions and start overlay
    calculate_display_dimensions()
    start_gaze_overlay()
    
    # Set global debug_calibration flag
    global debug_calibration
    debug_calibration = debug_calibration_flag
    
    debug_mode_on = False
    
    temp_center = (0,0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip camera horizontally to fix mirroring (if enabled)
        global FLIP_CAMERA_HORIZONTAL
        if FLIP_CAMERA_HORIZONTAL:
            frame = cv2.flip(frame, 1)  # 1 = horizontal flip

        # Crop and resize frame
        frame = crop_to_aspect_ratio(frame)

        #find the darkest point
        darkest_point = get_darkest_area(frame)

        if debug_mode_on:
            darkest_image = frame.copy()
            cv2.circle(darkest_image, darkest_point, 10, (0, 0, 255), -1)
            cv2.imshow('Darkest image patch', darkest_image)

        # Convert to grayscale to handle pixel value operations
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        darkest_pixel_value = gray_frame[darkest_point[1], darkest_point[0]]
        
        # apply thresholding operations at different levels
        # at least one should give us a good ellipse segment
        thresholded_image_strict = apply_binary_threshold(gray_frame, darkest_pixel_value, 5)#lite
        thresholded_image_strict = mask_outside_square(thresholded_image_strict, darkest_point, 250)

        thresholded_image_medium = apply_binary_threshold(gray_frame, darkest_pixel_value, 15)#medium
        thresholded_image_medium = mask_outside_square(thresholded_image_medium, darkest_point, 250)
        
        thresholded_image_relaxed = apply_binary_threshold(gray_frame, darkest_pixel_value, 25)#heavy
        thresholded_image_relaxed = mask_outside_square(thresholded_image_relaxed, darkest_point, 250)
        
        #take the three images thresholded at different levels and process them
        pupil_rotated_rect, pupil_center = process_frames(thresholded_image_strict, thresholded_image_medium, thresholded_image_relaxed, frame, gray_frame, darkest_point, debug_mode_on, True, debug_calibration_flag)
        
        # Update accuracy testing target position
        global accuracy_testing_active
        if accuracy_testing_active:
            update_accuracy_target()
            # Record accuracy measurement
            record_accuracy_measurement(current_gaze_x, current_gaze_y)
        
        # Update gaze overlay if enabled - directly call update from video loop
        if overlay_running and current_gaze_x is not None and current_gaze_y is not None:
            # Direct update from video loop - this ensures updates happen
            try:
                # Call update directly instead of relying on periodic callback
                if gaze_overlay_window is not None and gaze_overlay_canvas is not None:
                    update_gaze_overlay(current_gaze_x, current_gaze_y)
                    # Also process tkinter events
                    gaze_overlay_window.update_idletasks()
            except Exception as e:
                # Silently fail to avoid interrupting video processing
                pass
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('d') and debug_mode_on == False:  # Press 'd' to start debug mode
            debug_mode_on = True
        elif key == ord('d') and debug_mode_on == True:
            debug_mode_on = False
            cv2.destroyAllWindows()
        
        # Calibration: look at screen center and press 'c'
        if key == ord('c'):
            if raw_gaze_x is not None and raw_gaze_y is not None:
                calibrate_gaze(raw_gaze_x, raw_gaze_y)
            else:
                print("No gaze data available for calibration. Make sure pupil is detected.")
        
        # Toggle gaze overlay
        if key == ord('g'):
            if overlay_running:
                stop_gaze_overlay()
            else:
                start_gaze_overlay()
        
        # Start/stop accuracy testing
        if key == ord('a'):
            if not accuracy_testing_active:
                if not overlay_running:
                    print("Please enable gaze overlay (press 'g') before starting accuracy testing.")
                else:
                    start_accuracy_testing(grid_size=3)
            else:
                stop_accuracy_testing()
        
        if key == ord('q'):  # Press 'q' to quit
            stop_gaze_overlay()
            out.release()
            break   
        elif key == ord(' '):  # Press spacebar to start/stop
            while True:
                key = cv2.waitKey(1) & 0xFF
                if key == ord(' '):  # Press spacebar again to resume
                    break
                elif key == ord('q'):  # Press 'q' to quit
                    break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

#Prompts the user to select a video file if the hardcoded path is not found
#This is just for my debugging convenience :)
def select_video(debug_calibration_flag=False):
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    video_path = 'C:/Google Drive/Eye Tracking/fulleyetest.mp4'
    if not os.path.exists(video_path):
        print("No file found at hardcoded path. Please select a video file.")
        video_path = filedialog.askopenfilename(title="Select Video File", filetypes=[("Video Files", "*.mp4")])
        if not video_path:
            print("No file selected. Exiting.")
            return
            
    #second parameter is 1 for video 2 for webcam
    process_video(video_path, 2, debug_calibration_flag)

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Eye Tracking Phygital Application')
    parser.add_argument('--debug', action='store_true', 
                       help='Enable debug calibration mode with visual ovals at key points')
    args = parser.parse_args()
    
    if args.debug:
        print("🔍 Debug calibration mode enabled")
        print("   Visual ovals will be shown at:")
        print("   - Pupil position (magenta on frame)")
        print("   - Frame center (yellow on frame)")
        print("   - Gaze position (cyan on overlay)")
        print("   - Screen center (green on overlay)")
        print("")
    
    select_video(debug_calibration_flag=args.debug)


