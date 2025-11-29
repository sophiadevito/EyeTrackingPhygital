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
PLR_FLASH_DURATION_SEC = 2.5  # Duration of white flash for PLR test

# Store samples for calibration
calibration_samples = []  # List of (timestamp, target_x, target_y, pupil_x, pupil_y, raw_gaze_x, raw_gaze_y)

# PLR test globals
plr_test_active = False
plr_baseline_samples = []  # Pupil diameter samples before light flash (baseline)
plr_response_samples = []  # Pupil diameter samples during/after light flash
plr_flash_start_time = None
plr_baseline_start_time = None
plr_baseline_duration_sec = 1.0  # Collect baseline for 1 second before flash
showing_plr_flash = False
plr_flash_window = None
plr_flash_canvas = None

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
    """Stop calibration and calculate transformation matrix, then start PLR test"""
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
            
            # Start PLR test after calibration
            print("\nStarting PLR (Pupillary Light Reflex) test...")
            start_plr_test()
        
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
        global plr_test_active, showing_plr_flash
        
        # Don't draw calibration overlay during PLR flash (separate white window)
        if plr_test_active and showing_plr_flash:
            return
        
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

# ------------------- PLR Test Functions -------------------

def create_plr_flash_window():
    """Create fullscreen white flash window for PLR test"""
    global plr_flash_window, plr_flash_canvas, monitor_width, monitor_height
    
    if monitor_width is None or monitor_height is None:
        get_monitor_resolution()
    
    try:
        import tkinter as tk
        import sys
        
        root = tk.Tk()
        root.title("PLR Flash")
        root.attributes('-fullscreen', True)
        root.attributes('-topmost', True)
        root.configure(bg='white')
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
                          bg='white', highlightthickness=0)
        canvas.pack(fill=tk.BOTH, expand=True)
        
        plr_flash_window = root
        plr_flash_canvas = canvas
        
        return root, canvas
    except Exception as e:
        print(f"Error creating PLR flash window: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def start_plr_test():
    """Start PLR (Pupillary Light Reflex) test"""
    global plr_test_active, plr_baseline_samples, plr_response_samples
    global plr_flash_start_time, plr_baseline_start_time, showing_plr_flash
    global monitor_width, monitor_height
    
    if monitor_width is None or monitor_height is None:
        get_monitor_resolution()
    
    # Initialize PLR test
    plr_test_active = True
    plr_baseline_samples = []
    plr_response_samples = []
    plr_baseline_start_time = time.time()
    plr_flash_start_time = None
    showing_plr_flash = False
    
    print("  Collecting baseline pupil diameter (1 second)...")
    print("  Then screen will flash white for 2.5 seconds")
    print("  Keep your eyes open and look at the screen")

def stop_plr_test():
    """Stop PLR test and calculate results"""
    global plr_test_active, plr_baseline_samples, plr_response_samples
    global plr_flash_window, plr_flash_canvas, showing_plr_flash
    
    if not plr_test_active:
        return None
    
    plr_test_active = False
    showing_plr_flash = False
    
    # Close flash window
    if plr_flash_window is not None:
        try:
            plr_flash_window.destroy()
        except:
            pass
        plr_flash_window = None
        plr_flash_canvas = None
    
    if len(plr_baseline_samples) == 0 or len(plr_response_samples) == 0:
        print("Insufficient PLR data collected.")
        return None
    
    try:
        # Calculate PLR metrics
        result = calculate_plr_metrics(plr_baseline_samples, plr_response_samples)
        
        # Save PLR data
        save_plr_data(plr_baseline_samples, plr_response_samples, result)
        
        # Print results
        print_plr_results(result)
        
        return result
    except Exception as e:
        print(f"Error completing PLR test: {e}")
        import traceback
        traceback.print_exc()
        return None

def update_plr_test(pupil_diameter):
    """Update PLR test state - called from main loop"""
    global plr_test_active, plr_baseline_samples, plr_response_samples
    global plr_flash_start_time, plr_baseline_start_time, showing_plr_flash
    global plr_flash_window, plr_flash_canvas
    
    if not plr_test_active:
        return
    
    try:
        current_time = time.time()
        current_time_ms = current_time * 1000
        
        # Phase 1: Collect baseline (1 second before flash)
        if plr_flash_start_time is None:
            elapsed = current_time - plr_baseline_start_time
            
            if elapsed >= plr_baseline_duration_sec:
                # Start flash
                plr_flash_start_time = current_time
                showing_plr_flash = True
                
                # Create white flash window
                root, canvas = create_plr_flash_window()
                if root is not None:
                    # Process tkinter events to show window
                    try:
                        root.update_idletasks()
                        root.update()
                    except:
                        pass
                    
                    print("  Flash started!")
            else:
                # Still collecting baseline
                if pupil_diameter is not None:
                    plr_baseline_samples.append((current_time_ms, pupil_diameter))
        
        # Phase 2: During flash (collect response data)
        elif showing_plr_flash:
            elapsed = current_time - plr_flash_start_time
            
            # Record pupil diameter during flash
            if pupil_diameter is not None:
                plr_response_samples.append((current_time_ms, pupil_diameter))
            
            # Check if flash duration exceeded
            if elapsed >= PLR_FLASH_DURATION_SEC:
                # Stop PLR test
                stop_plr_test()
                return
    except Exception as e:
        print(f"Error in update_plr_test: {e}")
        import traceback
        traceback.print_exc()

def calculate_plr_metrics(baseline_samples, response_samples):
    """
    Calculate PLR metrics from baseline and response samples.
    
    Args:
        baseline_samples: List of (timestamp_ms, diameter) tuples before flash
        response_samples: List of (timestamp_ms, diameter) tuples during/after flash
    
    Returns:
        Dictionary with PLR metrics
    """
    if len(baseline_samples) == 0 or len(response_samples) == 0:
        return None
    
    # Calculate baseline diameter (average of last 0.5 seconds before flash)
    baseline_window_ms = 500
    if len(baseline_samples) > 0:
        last_timestamp = baseline_samples[-1][0]
        recent_baseline = [(t, d) for t, d in baseline_samples 
                          if last_timestamp - t <= baseline_window_ms]
        if len(recent_baseline) > 0:
            baseline_diameters = [d for _, d in recent_baseline]
            baseline_diameter = np.mean(baseline_diameters)
        else:
            baseline_diameters = [d for _, d in baseline_samples]
            baseline_diameter = np.mean(baseline_diameters)
    else:
        return None
    
    # Find minimum diameter during response (maximum constriction)
    response_diameters = [d for _, d in response_samples]
    min_response_diameter = np.min(response_diameters)
    
    # Calculate constriction amplitude (baseline - minimum)
    constriction_amplitude = baseline_diameter - min_response_diameter
    
    # Calculate PLR latency (time from flash start to constriction start)
    flash_start_time = response_samples[0][0] if response_samples else None
    if flash_start_time is None:
        return None
    
    # Detect when constriction starts (pupil diameter starts decreasing significantly)
    # Use a threshold: constriction starts when diameter drops by 5% from baseline
    constriction_threshold = baseline_diameter * 0.95  # 5% decrease
    
    latency_ms = None
    for timestamp, diameter in response_samples:
        if diameter <= constriction_threshold:
            latency_ms = timestamp - flash_start_time
            break
    
    # If no clear constriction detected, use first significant drop (2% decrease)
    if latency_ms is None:
        weak_threshold = baseline_diameter * 0.98  # 2% decrease
        for timestamp, diameter in response_samples:
            if diameter <= weak_threshold:
                latency_ms = timestamp - flash_start_time
                break
    
    # If still no detection, use time to minimum diameter
    if latency_ms is None and len(response_samples) > 0:
        min_index = np.argmin(response_diameters)
        if min_index > 0:
            latency_ms = response_samples[min_index][0] - flash_start_time
    
    result = {
        'baseline_diameter_pixels': float(baseline_diameter),
        'min_response_diameter_pixels': float(min_response_diameter),
        'constriction_amplitude_pixels': float(constriction_amplitude),
        'constriction_amplitude_percent': float((constriction_amplitude / baseline_diameter) * 100.0) if baseline_diameter > 0 else 0.0,
        'plr_latency_ms': float(latency_ms) if latency_ms is not None else None,
        'baseline_samples_count': len(baseline_samples),
        'response_samples_count': len(response_samples)
    }
    
    return result

def save_plr_data(baseline_samples, response_samples, result):
    """Save PLR data to JSON file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    json_filename = f"plr_results_{timestamp}.json"
    try:
        global monitor_width, monitor_height
        json_data = {
            'metadata': {
                'timestamp': timestamp,
                'test_date': datetime.now().isoformat(),
                'monitor_width': monitor_width,
                'monitor_height': monitor_height,
                'flash_duration_seconds': PLR_FLASH_DURATION_SEC,
                'baseline_duration_seconds': plr_baseline_duration_sec
            },
            'results': result,
            'baseline_data': [{'timestamp_ms': t, 'pupil_diameter_pixels': d} for t, d in baseline_samples],
            'response_data': [{'timestamp_ms': t, 'pupil_diameter_pixels': d} for t, d in response_samples]
        }
        with open(json_filename, 'w') as jsonfile:
            json.dump(json_data, jsonfile, indent=2)
        print(f"✓ PLR results saved to {json_filename}")
    except Exception as e:
        print(f"✗ Error saving PLR JSON: {e}")
        import traceback
        traceback.print_exc()

def print_plr_results(result):
    """Print PLR results to console"""
    if result is None:
        return
    
    print("\n" + "="*60)
    print("PLR (PUPILLARY LIGHT REFLEX) TEST RESULTS")
    print("="*60)
    print(f"Baseline Samples: {result['baseline_samples_count']}")
    print(f"Response Samples: {result['response_samples_count']}")
    print(f"\nPupil Diameter:")
    print(f"  Baseline:       {result['baseline_diameter_pixels']:.2f} pixels")
    print(f"  Minimum:        {result['min_response_diameter_pixels']:.2f} pixels")
    print(f"\nConstriction Amplitude:")
    print(f"  Absolute:       {result['constriction_amplitude_pixels']:.2f} pixels")
    print(f"  Percentage:     {result['constriction_amplitude_percent']:.2f}%")
    if result['plr_latency_ms'] is not None:
        print(f"\nPLR Latency:")
        print(f"  Time to constriction: {result['plr_latency_ms']:.2f} ms")
    else:
        print(f"\nPLR Latency: Could not be determined")
    print(f"{'='*60}\n")


