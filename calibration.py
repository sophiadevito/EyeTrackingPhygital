import numpy as np
import math
import time
import json
from datetime import datetime

# Calibration testing globals
calibration_active = False
calibration_data = []  # List to store calibration measurements
current_target_position = None  # Current target position (x, y)
target_index = 0
total_targets = 9  # 3x3 grid
showing_instruction = False
instruction_start_time = None
showing_target = False
target_start_time = None

# Timing parameters
INSTRUCTION_DURATION_MS = 3000  # How long to show instruction (ms)
TARGET_DURATION_SEC = 3.0  # Duration per target (seconds) - 9 targets * 3s = 27s, plus instruction = ~30s
# Store samples for calibration
calibration_samples = []  # List of (timestamp, target_x, target_y, pupil_x, pupil_y, raw_gaze_x, raw_gaze_y)

monitor_width = None
monitor_height = None
display_distance_mm = 400

def get_monitor_resolution():
    """Get primary monitor resolution - import from eye_tracking module"""
    import eye_tracking
    global monitor_width, monitor_height, display_distance_mm
    monitor_width, monitor_height = eye_tracking.monitor_width, eye_tracking.monitor_height
    # Always use 40cm = 400mm as the display distance
    if display_distance_mm is None or display_distance_mm <= 0:
        display_distance_mm = 400.0  # 40cm = 400mm
    return monitor_width, monitor_height

def generate_calibration_grid(grid_size=3, margin_percent=0.2):
    """
    Generate a 3x3 grid of calibration target positions.
    
    Args:
        grid_size: Number of targets per row/column (default 3x3 = 9 targets)
        margin_percent: Percentage of screen to leave as margin (default 20%)
    
    Returns:
        List of (x, y) tuples representing target positions
    """
    global monitor_width, monitor_height
    
    if monitor_width is None or monitor_height is None:
        get_monitor_resolution()
    
    if monitor_width is None or monitor_height is None:
        print("Error: Could not get monitor resolution")
        return []
    
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

def start_calibration():
    """Start 9-point calibration"""
    global calibration_active, target_index, calibration_data, calibration_samples
    global showing_instruction, instruction_start_time, showing_target, target_start_time
    global current_target_position, monitor_width, monitor_height
    
    if monitor_width is None or monitor_height is None:
        get_monitor_resolution()
    
    if monitor_width is None or monitor_height is None:
        print("Error: Could not get monitor resolution for calibration")
        return False
    
    # Initialize calibration
    target_index = 0
    calibration_data = []
    calibration_samples = []
    calibration_active = True
    showing_instruction = True
    showing_target = False
    instruction_start_time = time.time()
    target_start_time = None
    current_target_position = None
    
    print("9-point calibration started!")
    print("  Instruction will be shown first")
    print("  Then 9 targets will appear (3x3 grid)")
    print("  Look at each target as it appears")
    print("  Keep your head still")
    print("  Press 'x' again to stop early")
    
    return True

def stop_calibration():
    """Stop calibration and calculate transformation matrix"""
    global calibration_active, calibration_samples, calibration_data
    
    if not calibration_active:
        return None
    
    calibration_active = False
    showing_instruction = False
    showing_target = False
    
    if len(calibration_data) == 0:
        print("No calibration data collected.")
        return None
    
    try:
        # Calculate calibration matrix
        result = calculate_calibration_matrix(calibration_data)
        
        if result is not None:
            # Apply calibration to eye_tracking module
            apply_calibration(result)
            
            # Save calibration data
            save_calibration_data(calibration_data, result)
            
            # Print results
            print_calibration_results(result)
        
        return result
    except Exception as e:
        print(f"Error completing calibration: {e}")
        import traceback
        traceback.print_exc()
        return None

def calculate_calibration_matrix(data):
    """
    Calculate calibration transformation matrix from collected data.
    
    Uses least squares to find a 2D affine transformation that maps
    raw gaze positions to target positions.
    
    Returns:
        Dictionary with calibration parameters
    """
    if len(data) == 0:
        return None
    
    # Extract target and raw gaze positions
    targets = np.array([(d['target_x'], d['target_y']) for d in data])
    raw_gazes = np.array([(d['raw_gaze_x'], d['raw_gaze_y']) for d in data])
    
    # Calculate means for centering
    target_mean = np.mean(targets, axis=0)
    raw_gaze_mean = np.mean(raw_gazes, axis=0)
    
    # Center the data
    targets_centered = targets - target_mean
    raw_gazes_centered = raw_gazes - raw_gaze_mean
    
    # Calculate scaling and rotation using least squares
    # We want to find transformation: target = scale * rotation * raw_gaze + offset
    
    # Calculate covariance matrix
    cov_matrix = np.dot(raw_gazes_centered.T, targets_centered)
    
    # SVD to find optimal rotation
    U, S, Vt = np.linalg.svd(cov_matrix)
    
    # Rotation matrix
    R = np.dot(Vt.T, U.T)
    
    # Ensure proper rotation (det(R) should be 1, but we allow reflection)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = np.dot(Vt.T, U.T)
    
    # Calculate scaling factors (separate for X and Y)
    # Solve: targets = scale_x * raw_gazes_x + scale_y * raw_gazes_y + offset
    # We'll use linear regression for each dimension
    
    # For X dimension
    X_raw = raw_gazes_centered[:, 0]
    X_target = targets_centered[:, 0]
    scale_x = np.dot(X_raw, X_target) / np.dot(X_raw, X_raw) if np.dot(X_raw, X_raw) > 0 else 1.0
    
    # For Y dimension
    Y_raw = raw_gazes_centered[:, 1]
    Y_target = targets_centered[:, 1]
    scale_y = np.dot(Y_raw, Y_target) / np.dot(Y_raw, Y_raw) if np.dot(Y_raw, Y_raw) > 0 else 1.0
    
    # Calculate offset
    offset = target_mean - (scale_x * raw_gaze_mean[0], scale_y * raw_gaze_mean[1])
    
    # Alternative: Use simple offset (mean difference)
    simple_offset = target_mean - raw_gaze_mean
    
    # Calculate error metrics
    # Apply transformation to raw gazes
    transformed_x = raw_gazes[:, 0] * scale_x + offset[0]
    transformed_y = raw_gazes[:, 1] * scale_y + offset[1]
    transformed = np.column_stack([transformed_x, transformed_y])
    
    errors = np.sqrt(np.sum((transformed - targets)**2, axis=1))
    mean_error = np.mean(errors)
    max_error = np.max(errors)
    
    result = {
        'scale_x': float(scale_x),
        'scale_y': float(scale_y),
        'offset_x': float(offset[0]),
        'offset_y': float(offset[1]),
        'simple_offset_x': float(simple_offset[0]),
        'simple_offset_y': float(simple_offset[1]),
        'target_mean_x': float(target_mean[0]),
        'target_mean_y': float(target_mean[1]),
        'raw_gaze_mean_x': float(raw_gaze_mean[0]),
        'raw_gaze_mean_y': float(raw_gaze_mean[1]),
        'mean_error_pixels': float(mean_error),
        'max_error_pixels': float(max_error),
        'num_points': len(data)
    }
    
    return result

def apply_calibration(calibration_result):
    """
    Apply calibration result to eye_tracking module.
    
    Updates the calibration offsets in eye_tracking.py
    """
    import eye_tracking
    
    # Apply both offset and scale
    eye_tracking.calibration_offset_x = calibration_result['offset_x']
    eye_tracking.calibration_offset_y = calibration_result['offset_y']
    
    # Store calibration scale parameters
    eye_tracking.calibration_scale_x = calibration_result['scale_x']
    eye_tracking.calibration_scale_y = calibration_result['scale_y']
    
    print(f"Calibration applied!")
    print(f"  Offset: ({calibration_result['offset_x']:.1f}, {calibration_result['offset_y']:.1f})")
    print(f"  Scale: ({calibration_result['scale_x']:.3f}, {calibration_result['scale_y']:.3f})")
    print(f"  Mean error: {calibration_result['mean_error_pixels']:.2f} pixels")
    print(f"  Max error: {calibration_result['max_error_pixels']:.2f} pixels")

def update_calibration(gaze_x, gaze_y, raw_gaze_x, raw_gaze_y, pupil_x, pupil_y):
    """Update calibration state - called from main loop"""
    global showing_instruction, instruction_start_time
    global showing_target, target_start_time, target_index, total_targets
    global calibration_data, current_target_position
    global monitor_width, monitor_height
    
    if not calibration_active:
        return
    
    try:
        # Ensure monitor resolution is available
        if monitor_width is None or monitor_height is None:
            get_monitor_resolution()
        
        current_time = time.time()
        
        # State machine: instruction -> targets -> done
        
        if showing_instruction:
            elapsed_ms = (current_time - instruction_start_time) * 1000
            if elapsed_ms >= INSTRUCTION_DURATION_MS:
                # Move to showing targets
                showing_instruction = False
                showing_target = True
                target_index = 0
                
                # Generate calibration grid
                target_positions = generate_calibration_grid(3, 0.2)
                if not hasattr(update_calibration, 'target_positions'):
                    update_calibration.target_positions = target_positions
                
                if len(target_positions) > 0:
                    current_target_position = target_positions[0]
                    target_start_time = current_time
        
        elif showing_target:
            if not hasattr(update_calibration, 'target_positions'):
                # No targets generated, stop
                stop_calibration()
                return
            
            elapsed = current_time - target_start_time
            
            # Record calibration sample while target is shown
            if (gaze_x is not None and gaze_y is not None and 
                raw_gaze_x is not None and raw_gaze_y is not None and
                current_target_position is not None):
                
                target_x, target_y = current_target_position
                
                # Record sample (only once per target, after 1 second of stabilization)
                if elapsed >= 1.0 and elapsed <= 2.5:
                    # Only record once per target (check if we already recorded this target)
                    if not hasattr(update_calibration, 'recorded_targets'):
                        update_calibration.recorded_targets = set()
                    
                    if target_index not in update_calibration.recorded_targets:
                        update_calibration.recorded_targets.add(target_index)
                        
                        measurement = {
                            'target_index': target_index,
                            'target_x': target_x,
                            'target_y': target_y,
                            'raw_gaze_x': raw_gaze_x,
                            'raw_gaze_y': raw_gaze_y,
                            'pupil_x': pupil_x if pupil_x is not None else 0,
                            'pupil_y': pupil_y if pupil_y is not None else 0,
                            'timestamp': current_time
                        }
                        calibration_data.append(measurement)
            
            # Check if target duration exceeded
            if elapsed >= TARGET_DURATION_SEC:
                target_index += 1
                
                if target_index >= len(update_calibration.target_positions):
                    # All targets visited, complete calibration
                    stop_calibration()
                    return
                else:
                    # Move to next target
                    current_target_position = update_calibration.target_positions[target_index]
                    target_start_time = current_time
    except Exception as e:
        print(f"Error in update_calibration: {e}")
        import traceback
        traceback.print_exc()
        # Don't stop the calibration, just log the error

def save_calibration_data(data, result):
    """Save calibration data to JSON file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    json_filename = f"calibration_results_{timestamp}.json"
    try:
        global monitor_width, monitor_height
        json_data = {
            'metadata': {
                'timestamp': timestamp,
                'calibration_date': datetime.now().isoformat(),
                'monitor_width': monitor_width,
                'monitor_height': monitor_height,
                'num_points': len(data)
            },
            'calibration_parameters': result,
            'raw_data': data
        }
        with open(json_filename, 'w') as jsonfile:
            json.dump(json_data, jsonfile, indent=2)
        print(f"✓ Calibration results saved to {json_filename}")
    except Exception as e:
        print(f"✗ Error saving JSON: {e}")
        import traceback
        traceback.print_exc()

def print_calibration_results(result):
    """Print calibration results to console"""
    if result is None:
        return
    
    print("\n" + "="*60)
    print("9-POINT CALIBRATION RESULTS")
    print("="*60)
    print(f"Number of Calibration Points: {result['num_points']}")
    print(f"\nCalibration Parameters:")
    print(f"  Scale X:         {result['scale_x']:.4f}")
    print(f"  Scale Y:         {result['scale_y']:.4f}")
    print(f"  Offset X:        {result['offset_x']:.2f} pixels")
    print(f"  Offset Y:        {result['offset_y']:.2f} pixels")
    print(f"  Simple Offset X: {result['simple_offset_x']:.2f} pixels")
    print(f"  Simple Offset Y: {result['simple_offset_y']:.2f} pixels")
    print(f"\nError Metrics:")
    print(f"  Mean Error:      {result['mean_error_pixels']:.2f} pixels")
    print(f"  Max Error:       {result['max_error_pixels']:.2f} pixels")
    print(f"{'='*60}\n")

def get_calibration_overlay_draw_function():
    """Returns a function that draws calibration elements on the overlay canvas"""
    def draw_on_overlay(canvas):
        global calibration_active, showing_instruction, showing_target
        global current_target_position, monitor_width, monitor_height, target_index, total_targets
        
        try:
            if not calibration_active:
                return
            
            # Ensure monitor resolution is available
            if monitor_width is None or monitor_height is None:
                get_monitor_resolution()
            
            if monitor_width is None or monitor_height is None:
                # Can't draw without monitor resolution
                return
            
            # Show instruction
            if showing_instruction:
                center_x = monitor_width // 2
                center_y = monitor_height // 2
                canvas.create_text(
                    center_x, center_y - 30,
                    text="Look at each of the dots as they appear on the screen.",
                    fill='white', font=('Arial', 22, 'bold'),
                    justify='center'
                )
                canvas.create_text(
                    center_x, center_y + 20,
                    text="Keep your head still.",
                    fill='white', font=('Arial', 22, 'bold'),
                    justify='center'
                )
            
            # Show calibration target
            elif showing_target:
                if current_target_position is not None:
                    try:
                        x, y = current_target_position
                        # Validate coordinates are within screen bounds
                        if x < 0 or x > monitor_width or y < 0 or y > monitor_height:
                            # Skip drawing if coordinates are invalid
                            return
                        
                        # Draw target circle (larger for calibration)
                        dot_radius = 15
                        color = 'cyan'  # Cyan for calibration target
                        canvas.create_oval(
                            x - dot_radius, y - dot_radius,
                            x + dot_radius, y + dot_radius,
                            fill=color, outline='white', width=2
                        )
                        # Draw inner dot
                        inner_radius = 5
                        canvas.create_oval(
                            x - inner_radius, y - inner_radius,
                            x + inner_radius, y + inner_radius,
                            fill='white', outline='white'
                        )
                        # Show progress
                        canvas.create_text(
                            x, y - dot_radius - 25,
                            text=f"{target_index + 1}/{total_targets}",
                            fill='cyan', font=('Arial', 16, 'bold')
                        )
                    except (TypeError, ValueError) as e:
                        # Handle invalid coordinates
                        print(f"Warning: Invalid target position: {current_target_position}")
                        pass
        except Exception as e:
            # Print error for debugging but don't crash
            print(f"Error in calibration overlay drawing: {e}")
            import traceback
            traceback.print_exc()
    
    return draw_on_overlay


