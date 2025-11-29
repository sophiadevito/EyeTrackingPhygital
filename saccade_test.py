import numpy as np
import math
import time
import json
import random
from datetime import datetime

# Saccade testing globals
saccade_testing_active = False
saccade_data = []  # List to store saccade measurements
antisaccade_data = []  # List to store antisaccade measurements
test_phase = 'normal'  # 'normal' or 'antisaccade'
current_target_position = None  # Current target position (x, y)
current_target_angle = None  # Target angle in degrees
target_start_time = None  # Time when target appeared
gap_start_time = None  # Time when gap period started
showing_instruction = False
instruction_start_time = None
showing_central_dot = False
showing_target = False
showing_gap = False
target_index = 0
total_targets = 12

# Saccade detection parameters
SACCADE_VELOCITY_THRESHOLD = 30.0  # degrees/s - minimum velocity to detect saccade
BASELINE_WINDOW_MS = 50  # ms before target appearance to establish baseline
SACCADE_DETECTION_WINDOW_MS = 500  # ms after target appearance to detect saccade

# Timing parameters
GAP_DURATION_MS = 200  # Gap duration in milliseconds
TARGET_DURATION_MS = 1000  # How long to show each target (ms) - normal saccade
ANTISACCADE_TARGET_DURATION_MS = 1000  # Same duration as normal saccade targets (ms)
INSTRUCTION_DURATION_MS = 3000  # How long to show instruction (ms)

# Store gaze samples for saccade detection
gaze_samples = []  # List of (timestamp, x, y) tuples
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

def calculate_screen_position_from_angle(angle_degrees):
    """
    Calculate screen position (in pixels) from viewing angle in degrees.
    
    Args:
        angle_degrees: Horizontal viewing angle in degrees (positive = right, negative = left)
    
    Returns:
        (x, y): Screen position in pixels, centered at screen center
    """
    global monitor_width, monitor_height, display_distance_mm
    
    try:
        if monitor_width is None or monitor_height is None:
            get_monitor_resolution()
        
        if monitor_width is None or monitor_height is None:
            print("Error: Could not get monitor resolution")
            return (monitor_width // 2 if monitor_width else 960, 
                    monitor_height // 2 if monitor_height else 540)
        
        # Always use 40cm = 400mm as the display distance
        if display_distance_mm is None or display_distance_mm <= 0:
            display_distance_mm = 400.0  # 40cm = 400mm
        
        if display_distance_mm <= 0:
            print("Error: Invalid display distance")
            return (monitor_width // 2, monitor_height // 2)
        
        # Convert angle to radians
        angle_rad = math.radians(angle_degrees)
        
        # Calculate offset in mm from center
        # tan(angle) = offset / distance
        offset_mm = math.tan(angle_rad) * display_distance_mm
        
        # Convert mm to pixels using display dimensions
        import eye_tracking
        if eye_tracking.display_width_mm is not None and eye_tracking.display_width_mm > 0:
            pixels_per_mm = monitor_width / eye_tracking.display_width_mm
        else:
            # Fallback: assume 96 DPI
            pixels_per_mm = 96.0 / 25.4  # pixels per mm at 96 DPI
        
        offset_pixels = offset_mm * pixels_per_mm
        
        # Center position on screen
        screen_x = monitor_width // 2 + int(offset_pixels)
        screen_y = monitor_height // 2  # Keep at vertical center
        
        # Clamp to screen bounds
        screen_x = max(0, min(monitor_width - 1, screen_x))
        screen_y = max(0, min(monitor_height - 1, screen_y))
        
        return screen_x, screen_y
    except Exception as e:
        print(f"Error calculating screen position from angle {angle_degrees}: {e}")
        import traceback
        traceback.print_exc()
        # Return center position as fallback
        return (monitor_width // 2 if monitor_width else 960, 
                monitor_height // 2 if monitor_height else 540)

def generate_target_angles(num_targets=12, min_angle=10, max_angle=30, antisaccade=False):
    """
    Generate random horizontal target angles.
    
    Args:
        num_targets: Number of targets to generate
        min_angle: Minimum angle in degrees (absolute value)
        max_angle: Maximum angle in degrees (absolute value)
        antisaccade: If True, include more angles closer to center (5-20 degrees)
    
    Returns:
        List of angles in degrees (mix of positive and negative)
    """
    targets = []
    
    if antisaccade:
        # For antisaccade, use smaller angles (5-20 degrees) with more close to center
        # Mix of close (5-12 degrees) and medium (12-20 degrees)
        for i in range(num_targets):
            # Randomly choose left or right
            direction = random.choice([-1, 1])
            # More targets closer to center (60% chance of 5-12 degrees, 40% chance of 12-20 degrees)
            if random.random() < 0.6:
                angle = direction * random.uniform(5, 12)
            else:
                angle = direction * random.uniform(12, 20)
            targets.append(angle)
    else:
        # Normal saccade: use original range
        for _ in range(num_targets):
            # Randomly choose left or right
            direction = random.choice([-1, 1])
            # Random angle between min and max
            angle = direction * random.uniform(min_angle, max_angle)
            targets.append(angle)
    
    # Shuffle to randomize order
    random.shuffle(targets)
    return targets

def start_saccade_test():
    """Start saccade testing"""
    global saccade_testing_active, target_index, total_targets
    global showing_instruction, instruction_start_time, showing_central_dot, showing_target, showing_gap
    global current_target_position, current_target_angle, target_start_time, gap_start_time
    global gaze_samples, saccade_data
    
    if monitor_width is None or monitor_height is None:
        get_monitor_resolution()
    
    if monitor_width is None or monitor_height is None:
        print("Error: Could not get monitor resolution for saccade testing")
        return False
    
    # Initialize test
    target_index = 0
    saccade_data = []
    antisaccade_data = []
    gaze_samples = []
    test_phase = 'normal'  # Start with normal saccade phase
    saccade_testing_active = True
    showing_instruction = True
    showing_central_dot = False
    showing_target = False
    showing_gap = False
    instruction_start_time = time.time()
    current_target_position = None
    current_target_angle = None
    
    # Clear any previous state
    if hasattr(update_saccade_test, 'central_dot_start_time'):
        delattr(update_saccade_test, 'central_dot_start_time')
    if hasattr(update_saccade_test, 'recorded_for_target'):
        delattr(update_saccade_test, 'recorded_for_target')
    if hasattr(update_saccade_test, 'target_angles'):
        delattr(update_saccade_test, 'target_angles')
    
    print("Saccade test started!")
    print("  Instruction will be shown first")
    print("  Then 12 targets will appear with gap paradigm")
    print("  After normal saccade test, antisaccade test will begin")
    print("  Press 's' again to stop early")
    
    return True

def stop_saccade_test():
    """Stop saccade testing and calculate results, or transition to antisaccade phase"""
    global saccade_testing_active, saccade_data, antisaccade_data, test_phase, target_index, total_targets
    global showing_instruction, instruction_start_time, showing_central_dot, showing_target, showing_gap
    global current_target_position, current_target_angle
    
    if not saccade_testing_active:
        return None
    
    # If we're in normal phase and just finished, transition to antisaccade
    if test_phase == 'normal':
        if len(saccade_data) == 0:
            print("No saccade data collected.")
            saccade_testing_active = False
            return None
        
        # Transition to antisaccade phase
        test_phase = 'antisaccade'
        target_index = 0
        antisaccade_data = []
        showing_instruction = True
        showing_central_dot = False
        showing_target = False
        showing_gap = False
        instruction_start_time = time.time()
        current_target_position = None
        current_target_angle = None
        
        # Clear state for antisaccade phase
        if hasattr(update_saccade_test, 'central_dot_start_time'):
            delattr(update_saccade_test, 'central_dot_start_time')
        if hasattr(update_saccade_test, 'recorded_for_target'):
            delattr(update_saccade_test, 'recorded_for_target')
        if hasattr(update_saccade_test, 'target_angles'):
            delattr(update_saccade_test, 'target_angles')
        
        print("\nNormal saccade test complete!")
        print("Starting antisaccade test...")
        print("  Instruction will be shown first")
        print("  Then 12 targets will appear")
        print("  Look AWAY from the targets (opposite side)")
        return None  # Continue with antisaccade phase
    
    # If we're in antisaccade phase, finish the test
    saccade_testing_active = False
    
    if len(saccade_data) == 0 and len(antisaccade_data) == 0:
        print("No saccade data collected.")
        return None
    
    try:
        # Calculate metrics for both phases
        normal_results = calculate_saccade_metrics(saccade_data) if len(saccade_data) > 0 else None
        antisaccade_results = calculate_antisaccade_metrics(antisaccade_data) if len(antisaccade_data) > 0 else None
        
        # Combine results
        combined_results = {
            'normal_saccade': normal_results,
            'antisaccade': antisaccade_results
        }
        
        # Save data
        save_saccade_data(saccade_data, antisaccade_data, combined_results)
        
        # Print results
        print_saccade_results(combined_results)
        
        return combined_results
    except Exception as e:
        print(f"Error stopping saccade test: {e}")
        import traceback
        traceback.print_exc()
        return None

def update_saccade_test(gaze_x, gaze_y):
    """Update saccade testing state - called from main loop"""
    global showing_instruction, instruction_start_time
    global showing_central_dot, showing_target, showing_gap
    global current_target_position, current_target_angle, target_start_time, gap_start_time
    global target_index, total_targets
    global gaze_samples, saccade_data, antisaccade_data, monitor_width, monitor_height, test_phase
    
    if not saccade_testing_active:
        return
    
    try:
        # Ensure monitor resolution is available
        if monitor_width is None or monitor_height is None:
            get_monitor_resolution()
        
        current_time = time.time()
        current_time_ms = current_time * 1000
        
        # Record gaze sample
        if gaze_x is not None and gaze_y is not None:
            gaze_samples.append((current_time_ms, gaze_x, gaze_y))
            # Keep only last 2 seconds of samples
            cutoff_time = current_time_ms - 2000
            gaze_samples = [(t, x, y) for t, x, y in gaze_samples if t > cutoff_time]
        
        # State machine: instruction -> central dot -> gap -> target -> next target
        
        if showing_instruction:
            elapsed_ms = (current_time - instruction_start_time) * 1000
            if elapsed_ms >= INSTRUCTION_DURATION_MS:
                # Move to showing central dot
                showing_instruction = False
                showing_central_dot = True
                current_target_position = (monitor_width // 2, monitor_height // 2)
                target_index = 0
                # Generate target angles - use different angles for antisaccade (closer to center)
                if not hasattr(update_saccade_test, 'target_angles'):
                    if test_phase == 'normal':
                        target_angles = generate_target_angles(total_targets, 10, 30, antisaccade=False)
                    else:
                        target_angles = generate_target_angles(total_targets, 5, 20, antisaccade=True)
                    update_saccade_test.target_angles = target_angles
        
        elif showing_central_dot:
            # Show central dot for a random duration (500-1000ms) before gap
            if not hasattr(update_saccade_test, 'central_dot_start_time'):
                update_saccade_test.central_dot_start_time = current_time
                update_saccade_test.central_dot_duration = random.uniform(0.5, 1.0)  # seconds
            
            elapsed = current_time - update_saccade_test.central_dot_start_time
            if elapsed >= update_saccade_test.central_dot_duration:
                # Move to gap
                showing_central_dot = False
                showing_gap = True
                gap_start_time = current_time
                current_target_position = None
        
        elif showing_gap:
            # Gap period (200ms)
            elapsed_ms = (current_time - gap_start_time) * 1000
            if elapsed_ms >= GAP_DURATION_MS:
                # Move to showing target
                showing_gap = False
                showing_target = True
                target_start_time = current_time
                
                # Get next target angle
                if hasattr(update_saccade_test, 'target_angles') and target_index < len(update_saccade_test.target_angles):
                    try:
                        current_target_angle = update_saccade_test.target_angles[target_index]
                        target_x, target_y = calculate_screen_position_from_angle(current_target_angle)
                        if target_x is not None and target_y is not None:
                            current_target_position = (target_x, target_y)
                            
                            # Clear gaze samples before target appears (for baseline)
                            baseline_cutoff = current_time_ms - BASELINE_WINDOW_MS
                            gaze_samples = [(t, x, y) for t, x, y in gaze_samples if t >= baseline_cutoff]
                        else:
                            print(f"Warning: Could not calculate target position for angle {current_target_angle}")
                            # Skip this target and move to next
                            target_index += 1
                            if target_index >= total_targets:
                                stop_saccade_test()
                                return
                    except Exception as e:
                        print(f"Error setting up target {target_index}: {e}")
                        import traceback
                        traceback.print_exc()
                        # Skip this target
                        target_index += 1
                        if target_index >= total_targets:
                            stop_saccade_test()
                            return
                else:
                    # All targets done
                    stop_saccade_test()
                    return
        
        elif showing_target:
            elapsed_ms = (current_time - target_start_time) * 1000
            
            # Detect saccade initiation (only once per target)
            if current_target_angle is not None and gaze_x is not None and gaze_y is not None:
                # Check if we've already recorded for this target
                if not hasattr(update_saccade_test, 'recorded_for_target'):
                    update_saccade_test.recorded_for_target = {}
                
                if target_index not in update_saccade_test.recorded_for_target:
                    if test_phase == 'normal':
                        # Normal saccade: detect movement towards target
                        saccade_latency = detect_saccade_initiation(target_start_time * 1000, current_target_angle)
                        
                        if saccade_latency is not None and saccade_latency > 0:
                            # Saccade detected, calculate velocity and accuracy
                            velocity = calculate_saccade_velocity(target_start_time * 1000, saccade_latency)
                            accuracy = calculate_saccade_accuracy(current_target_position, gaze_x, gaze_y, current_target_angle)
                            
                            # Record measurement
                            measurement = {
                                'target_index': target_index,
                                'target_angle_degrees': current_target_angle,
                                'target_x': current_target_position[0],
                                'target_y': current_target_position[1],
                                'saccade_latency_ms': saccade_latency,
                                'peak_velocity_deg_per_ms': velocity / 1000.0,  # Convert deg/s to deg/ms
                                'peak_velocity_deg_per_s': velocity,  # Also store in deg/s for reference
                                'accuracy_percent': accuracy
                            }
                            saccade_data.append(measurement)
                            update_saccade_test.recorded_for_target[target_index] = True
                            
                            # Move to next target or finish
                            target_index += 1
                            if target_index >= total_targets:
                                stop_saccade_test()
                                return
                            else:
                                # Reset for next target: show central dot again
                                showing_target = False
                                showing_central_dot = True
                                current_target_position = (monitor_width // 2, monitor_height // 2)
                                current_target_angle = None
                                # Clear attribute so it gets reset
                                if hasattr(update_saccade_test, 'central_dot_start_time'):
                                    delattr(update_saccade_test, 'central_dot_start_time')
                    else:
                        # Antisaccade: detect if eye moves towards target (error) or away (correct)
                        # Only record once per target (when first detected), but wait for full duration
                        if target_index not in update_saccade_test.recorded_for_target:
                            error = detect_antisaccade_error(target_start_time * 1000, current_target_angle)
                            
                            # Record measurement immediately when detected (early detection)
                            measurement = {
                                'target_index': target_index,
                                'target_angle_degrees': current_target_angle,
                                'target_x': current_target_position[0],
                                'target_y': current_target_position[1],
                                'error': error  # True if eye moved towards target (error), False if away (correct)
                            }
                            antisaccade_data.append(measurement)
                            update_saccade_test.recorded_for_target[target_index] = True
            
            # Duration check for antisaccade - needs to be outside the recorded_for_target check
            # so it runs every frame regardless of whether we've detected movement
            if test_phase == 'antisaccade' and showing_target:
                target_duration = ANTISACCADE_TARGET_DURATION_MS
                if elapsed_ms >= target_duration:
                    # Move to next target or finish
                    target_index += 1
                    if target_index >= total_targets:
                        stop_saccade_test()
                        return
                    else:
                        # Reset for next target: show central dot again
                        showing_target = False
                        showing_central_dot = True
                        current_target_position = (monitor_width // 2, monitor_height // 2)
                        current_target_angle = None
                        if hasattr(update_saccade_test, 'central_dot_start_time'):
                            delattr(update_saccade_test, 'central_dot_start_time')
            
            # Check if target duration exceeded (for cases where no detection happened)
            # Only check if we haven't already recorded and handled this target
            if test_phase == 'normal':
                # For normal saccade, use original timeout logic
                target_duration = TARGET_DURATION_MS
                if (target_index not in update_saccade_test.recorded_for_target and 
                    elapsed_ms >= target_duration):
                    # Mark as recorded (no saccade detected within time limit)
                    update_saccade_test.recorded_for_target[target_index] = True
                    
                    # Move to next target or finish
                    target_index += 1
                    if target_index >= total_targets:
                        stop_saccade_test()
                        return
                    else:
                        showing_target = False
                        showing_central_dot = True
                        current_target_position = (monitor_width // 2, monitor_height // 2)
                        current_target_angle = None
                        if hasattr(update_saccade_test, 'central_dot_start_time'):
                            delattr(update_saccade_test, 'central_dot_start_time')
            # For antisaccade, if no error was detected by the end of duration, record as no error
            # This handles the case where detect_antisaccade_error returned False and we never recorded
            elif test_phase == 'antisaccade' and showing_target:
                target_duration = ANTISACCADE_TARGET_DURATION_MS
                if (target_index not in update_saccade_test.recorded_for_target and 
                    elapsed_ms >= target_duration):
                    # No movement detected at all - record as correct (no error)
                    measurement = {
                        'target_index': target_index,
                        'target_angle_degrees': current_target_angle,
                        'target_x': current_target_position[0],
                        'target_y': current_target_position[1],
                        'error': False  # No movement = correct (no error)
                    }
                    antisaccade_data.append(measurement)
                    update_saccade_test.recorded_for_target[target_index] = True
                    
                    # Move to next target or finish
                    target_index += 1
                    if target_index >= total_targets:
                        stop_saccade_test()
                        return
                    else:
                        showing_target = False
                        showing_central_dot = True
                        current_target_position = (monitor_width // 2, monitor_height // 2)
                        current_target_angle = None
                        if hasattr(update_saccade_test, 'central_dot_start_time'):
                            delattr(update_saccade_test, 'central_dot_start_time')
    except Exception as e:
        print(f"Error in update_saccade_test: {e}")
        import traceback
        traceback.print_exc()
        # Don't stop the test, just log the error

def detect_saccade_initiation(target_time_ms, target_angle):
    """
    Detect when saccade starts (eye movement begins toward target).
    
    Args:
        target_time_ms: Time when target appeared (ms)
        target_angle: Target angle in degrees
    
    Returns:
        Saccade latency in ms, or None if not detected
    """
    global gaze_samples, display_distance_mm, monitor_width
    
    # Find samples within detection window
    window_start = target_time_ms
    window_end = target_time_ms + SACCADE_DETECTION_WINDOW_MS
    
    relevant_samples = [(t, x, y) for t, x, y in gaze_samples 
                       if window_start <= t <= window_end]
    
    if len(relevant_samples) < 2:
        return None
    
    # Calculate baseline gaze position (before target appeared)
    baseline_samples = [(t, x, y) for t, x, y in gaze_samples 
                       if target_time_ms - BASELINE_WINDOW_MS <= t < target_time_ms]
    
    if len(baseline_samples) == 0:
        return None
    
    baseline_x = np.mean([x for _, x, _ in baseline_samples])
    baseline_y = np.mean([y for _, _, y in baseline_samples])
    
    # Calculate target screen position
    target_x, target_y = calculate_screen_position_from_angle(target_angle)
    
    # Determine direction of movement (left or right)
    target_direction = 1 if target_angle > 0 else -1
    
    # Ensure monitor resolution is available
    if monitor_width is None:
        get_monitor_resolution()
    
    # Always use 40cm = 400mm as the display distance
    if display_distance_mm is None or display_distance_mm <= 0:
        display_distance_mm = 400.0  # 40cm = 400mm
    
    # Look for significant movement in target direction
    for i in range(1, len(relevant_samples)):
        t1, x1, y1 = relevant_samples[i-1]
        t2, x2, y2 = relevant_samples[i]
        
        # Calculate velocity (pixels per ms, convert to degrees/s)
        dt_ms = t2 - t1
        if dt_ms <= 0:
            continue
        
        dx_pixels = x2 - x1
        dy_pixels = y2 - y1
        
        # Calculate angular velocity
        # Convert pixel movement to angular movement
        import eye_tracking
        if eye_tracking.display_width_mm is not None:
            pixels_per_mm = monitor_width / eye_tracking.display_width_mm
        else:
            pixels_per_mm = 96.0 / 25.4
        
        dx_mm = dx_pixels / pixels_per_mm
        dy_mm = dy_pixels / pixels_per_mm
        
        # Calculate angle change
        distance_mm = math.sqrt(dx_mm**2 + dy_mm**2)
        
        angle_change_rad = math.atan2(distance_mm, display_distance_mm)
        angle_change_deg = math.degrees(angle_change_rad)
        
        # Velocity in degrees per second
        velocity_deg_per_s = (angle_change_deg / dt_ms) * 1000
        
        # Check if movement is in target direction and exceeds threshold
        movement_direction = 1 if dx_pixels > 0 else -1
        if movement_direction == target_direction and velocity_deg_per_s > SACCADE_VELOCITY_THRESHOLD:
            # Saccade initiated
            latency_ms = t1 - target_time_ms
            return max(0, latency_ms)  # Ensure non-negative
    
    return None

def calculate_saccade_velocity(target_time_ms, saccade_latency_ms):
    """
    Calculate peak angular velocity during saccade.
    
    Args:
        target_time_ms: Time when target appeared
        saccade_latency_ms: When saccade started (relative to target)
    
    Returns:
        Peak velocity in degrees/second
    """
    global gaze_samples, display_distance_mm, monitor_width
    
    # Look at samples during saccade (from initiation to 200ms after)
    saccade_start_ms = target_time_ms + saccade_latency_ms
    saccade_end_ms = saccade_start_ms + 200
    
    relevant_samples = [(t, x, y) for t, x, y in gaze_samples 
                       if saccade_start_ms <= t <= saccade_end_ms]
    
    if len(relevant_samples) < 2:
        return 0.0
    
    max_velocity = 0.0
    
    # Ensure monitor resolution is available
    if monitor_width is None:
        get_monitor_resolution()
    
    import eye_tracking
    if eye_tracking.display_width_mm is not None:
        pixels_per_mm = monitor_width / eye_tracking.display_width_mm
    else:
        pixels_per_mm = 96.0 / 25.4
    
    # Always use 40cm = 400mm as the display distance
    if display_distance_mm is None or display_distance_mm <= 0:
        display_distance_mm = 400.0  # 40cm = 400mm
    
    for i in range(1, len(relevant_samples)):
        t1, x1, y1 = relevant_samples[i-1]
        t2, x2, y2 = relevant_samples[i]
        
        dt_ms = t2 - t1
        if dt_ms <= 0:
            continue
        
        dx_pixels = x2 - x1
        dy_pixels = y2 - y1
        dx_mm = dx_pixels / pixels_per_mm
        dy_mm = dy_pixels / pixels_per_mm
        distance_mm = math.sqrt(dx_mm**2 + dy_mm**2)
        
        angle_change_rad = math.atan2(distance_mm, display_distance_mm)
        angle_change_deg = math.degrees(angle_change_rad)
        velocity_deg_per_s = (angle_change_deg / dt_ms) * 1000
        
        max_velocity = max(max_velocity, velocity_deg_per_s)
    
    return max_velocity

def calculate_saccade_accuracy(target_position, gaze_x, gaze_y, target_angle):
    """
    Calculate saccade accuracy (actual gaze / intended gaze).
    
    Args:
        target_position: (x, y) target position in pixels
        gaze_x, gaze_y: Actual gaze position in pixels
        target_angle: Target angle in degrees
    
    Returns:
        Accuracy as percentage (100% = perfect)
    """
    global display_distance_mm, monitor_width, monitor_height
    
    if target_position is None or gaze_x is None or gaze_y is None:
        return 0.0
    
    # Ensure monitor resolution is available
    if monitor_width is None or monitor_height is None:
        get_monitor_resolution()
    
    target_x, target_y = target_position
    
    # Calculate intended angle from center
    import eye_tracking
    center_x = monitor_width // 2
    center_y = monitor_height // 2
    
    # Calculate actual angle
    dx_pixels = gaze_x - center_x
    dy_pixels = gaze_y - center_y
    
    if eye_tracking.display_width_mm is not None:
        pixels_per_mm = monitor_width / eye_tracking.display_width_mm
    else:
        pixels_per_mm = 96.0 / 25.4
    
    dx_mm = dx_pixels / pixels_per_mm
    dy_mm = dy_pixels / pixels_per_mm
    distance_mm = math.sqrt(dx_mm**2 + dy_mm**2)
    
    # Always use 40cm = 400mm as the display distance
    if display_distance_mm is None or display_distance_mm <= 0:
        display_distance_mm = 400.0  # 40cm = 400mm
    
    actual_angle_rad = math.atan2(dx_mm, display_distance_mm)
    actual_angle_deg = math.degrees(actual_angle_rad)
    
    # Calculate accuracy: actual / intended * 100
    if abs(target_angle) < 0.1:  # Avoid division by zero
        return 100.0 if abs(actual_angle_deg) < 1.0 else 0.0
    
    # Use absolute values for accuracy calculation
    accuracy_percent = (abs(actual_angle_deg) / abs(target_angle)) * 100.0
    
    # Cap at reasonable values
    return min(200.0, max(0.0, accuracy_percent))

def detect_antisaccade_error(target_time_ms, target_angle):
    """
    Detect if eye moves towards target (error) or away from target (correct) in antisaccade test.
    Any movement in the direction of the target counts as an error.
    
    Args:
        target_time_ms: Time when target appeared (ms)
        target_angle: Target angle in degrees (positive = right, negative = left)
    
    Returns:
        True if error (eye moved towards target), False if correct (eye moved away or no movement)
    """
    global gaze_samples, display_distance_mm, monitor_width
    
    # Use longer detection window for antisaccade (match target duration)
    detection_window = ANTISACCADE_TARGET_DURATION_MS
    
    # Find samples within detection window
    window_start = target_time_ms
    window_end = target_time_ms + detection_window
    
    relevant_samples = [(t, x, y) for t, x, y in gaze_samples 
                       if window_start <= t <= window_end]
    
    if len(relevant_samples) < 2:
        return False  # No movement detected, assume correct (no error)
    
    # Calculate baseline gaze position (before target appeared)
    baseline_samples = [(t, x, y) for t, x, y in gaze_samples 
                       if target_time_ms - BASELINE_WINDOW_MS <= t < target_time_ms]
    
    if len(baseline_samples) == 0:
        return False
    
    baseline_x = np.mean([x for _, x, _ in baseline_samples])
    
    # Determine target direction (1 = right, -1 = left)
    target_direction = 1 if target_angle > 0 else -1
    
    # Ensure monitor resolution is available
    if monitor_width is None:
        get_monitor_resolution()
    
    # Look for ANY movement in target direction (lower threshold for antisaccade)
    # Any movement towards target = error
    MIN_MOVEMENT_PIXELS = 5  # Minimum pixel movement to detect direction
    
    for i in range(1, len(relevant_samples)):
        t1, x1, y1 = relevant_samples[i-1]
        t2, x2, y2 = relevant_samples[i]
        
        dx_pixels = x2 - x1
        
        # Check if there's any movement in the direction of the target
        if abs(dx_pixels) >= MIN_MOVEMENT_PIXELS:
            movement_direction = 1 if dx_pixels > 0 else -1
            
            # Error if movement is in same direction as target (towards target)
            if movement_direction == target_direction:
                return True  # Error: moved towards target
    
    # No movement towards target detected, assume correct (no error)
    return False

def calculate_saccade_metrics(data):
    """Calculate average metrics from collected saccade data"""
    if len(data) == 0:
        return None
    
    latencies = [d['saccade_latency_ms'] for d in data if d.get('saccade_latency_ms') is not None]
    velocities_deg_per_ms = [d.get('peak_velocity_deg_per_ms', d.get('peak_velocity_deg_per_s', 0) / 1000.0) 
                            for d in data if d.get('saccade_latency_ms') is not None]
    accuracies = [d['accuracy_percent'] for d in data if d.get('accuracy_percent') is not None]
    
    results = {
        'total_saccades': len(data),
        'valid_saccades': len(latencies),
        'average_latency_ms': np.mean(latencies) if latencies else 0.0,
        'average_velocity_deg_per_ms': np.mean(velocities_deg_per_ms) if velocities_deg_per_ms else 0.0,
        'average_accuracy_percent': np.mean(accuracies) if accuracies else 0.0,
        'std_latency_ms': np.std(latencies) if latencies else 0.0,
        'std_velocity_deg_per_ms': np.std(velocities_deg_per_ms) if velocities_deg_per_ms else 0.0,
        'std_accuracy_percent': np.std(accuracies) if accuracies else 0.0
    }
    
    return results

def calculate_antisaccade_metrics(data):
    """Calculate antisaccade error rate from collected data"""
    if len(data) == 0:
        return None
    
    errors = [d['error'] for d in data if 'error' in d]
    
    if len(errors) == 0:
        return None
    
    error_count = sum(errors)
    total_trials = len(errors)
    error_rate = (error_count / total_trials) * 100.0 if total_trials > 0 else 0.0
    
    results = {
        'total_trials': total_trials,
        'error_count': error_count,
        'correct_count': total_trials - error_count,
        'error_rate_percent': error_rate
    }
    
    return results

def save_saccade_data(normal_data, antisaccade_data, results):
    """Save saccade data to JSON file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    json_filename = f"saccade_results_{timestamp}.json"
    try:
        global monitor_width, monitor_height
        json_data = {
            'metadata': {
                'timestamp': timestamp,
                'test_date': datetime.now().isoformat(),
                'monitor_width': monitor_width,
                'monitor_height': monitor_height,
                'total_targets': total_targets,
                'gap_duration_ms': GAP_DURATION_MS
            },
            'results': results,
            'normal_saccade_data': normal_data,
            'antisaccade_data': antisaccade_data
        }
        with open(json_filename, 'w') as jsonfile:
            json.dump(json_data, jsonfile, indent=2)
        print(f"✓ Saccade results saved to {json_filename}")
    except Exception as e:
        print(f"✗ Error saving JSON: {e}")
        import traceback
        traceback.print_exc()

def print_saccade_results(results):
    """Print saccade results to console"""
    if results is None:
        return
    
    print("\n" + "="*60)
    print("SACCADE TEST RESULTS")
    print("="*60)
    
    # Normal saccade results
    if results.get('normal_saccade') is not None:
        normal = results['normal_saccade']
        print(f"\nNORMAL SACCADE TEST:")
        print(f"  Total Targets: {normal['total_saccades']}")
        print(f"  Valid Saccades Detected: {normal['valid_saccades']}")
        print(f"  Average Latency: {normal['average_latency_ms']:.2f} ms (std: {normal['std_latency_ms']:.2f})")
        print(f"  Average Velocity: {normal['average_velocity_deg_per_ms']:.4f} deg/ms (std: {normal['std_velocity_deg_per_ms']:.4f})")
        print(f"  Average Accuracy: {normal['average_accuracy_percent']:.2f}% (std: {normal['std_accuracy_percent']:.2f})")
    
    # Antisaccade results
    if results.get('antisaccade') is not None:
        anti = results['antisaccade']
        print(f"\nANTISACCADE TEST:")
        print(f"  Total Trials: {anti['total_trials']}")
        print(f"  Errors: {anti['error_count']}")
        print(f"  Correct: {anti['correct_count']}")
        print(f"  Average Error Rate: {anti['error_rate_percent']:.2f}%")
    
    print(f"{'='*60}\n")

def get_saccade_overlay_draw_function():
    """Returns a function that draws saccade testing elements on the overlay canvas"""
    def draw_on_overlay(canvas):
        global saccade_testing_active, showing_instruction, showing_central_dot
        global showing_target, current_target_position, monitor_width, monitor_height
        
        try:
            if not saccade_testing_active:
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
                global test_phase
                if test_phase == 'normal':
                    instruction_text = "When the dot appears, look at it as fast as you can"
                else:
                    instruction_text = "When the dot appears, look immediately to the opposite side"
                canvas.create_text(
                    center_x, center_y - 50,
                    text=instruction_text,
                    fill='white', font=('Arial', 24, 'bold'),
                    justify='center'
                )
            
            # Show central dot or target
            elif showing_central_dot or showing_target:
                if current_target_position is not None:
                    try:
                        x, y = current_target_position
                        # Validate coordinates are within screen bounds
                        if x < 0 or x > monitor_width or y < 0 or y > monitor_height:
                            # Skip drawing if coordinates are invalid
                            return
                        
                        dot_radius = 8 if showing_central_dot else 12
                        color = 'white' if showing_central_dot else 'green'
                        canvas.create_oval(
                            x - dot_radius, y - dot_radius,
                            x + dot_radius, y + dot_radius,
                            fill=color, outline=color
                        )
                    except (TypeError, ValueError) as e:
                        # Handle invalid coordinates
                        print(f"Warning: Invalid target position: {current_target_position}")
                        pass
        except Exception as e:
            # Print error for debugging but don't crash
            print(f"Error in saccade overlay drawing: {e}")
            import traceback
            traceback.print_exc()
    
    return draw_on_overlay

