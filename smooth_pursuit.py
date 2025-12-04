import numpy as np
import math
import time
import json
import random
from datetime import datetime

# Smooth pursuit testing globals
smooth_pursuit_active = False
pursuit_data = []  # List to store pursuit measurements
current_target_position = None  # Current target position (x, y)
test_start_time = None
showing_instruction = False
instruction_start_time = None
showing_target = False
target_start_time = None
phase = 'horizontal'  # 'horizontal' or 'vertical'
phase_start_time = None

# Timing parameters
INSTRUCTION_DURATION_MS = 3000  # How long to show instruction (ms)
PHASE_DURATION_SEC = 15.0  # Duration of each phase (horizontal/vertical) in seconds
FREQUENCY_MIN = 0.2  # Minimum frequency in Hz
FREQUENCY_MAX = 0.5  # Maximum frequency in Hz

# Store gaze samples for pursuit tracking
gaze_samples = []  # List of (timestamp, x, y) tuples
monitor_width = None
monitor_height = None
display_distance_mm = 400

# Target motion parameters
target_frequency = None  # Will be randomly selected
target_amplitude_deg = 15.0  # ±15 degrees
target_amplitude_pixels = None  # Will be calculated from degrees

def get_monitor_resolution():
    """Get primary monitor resolution - import from eye_tracking module"""
    import eye_tracking
    global monitor_width, monitor_height, display_distance_mm
    monitor_width, monitor_height = eye_tracking.monitor_width, eye_tracking.monitor_height
    # Always use 40cm = 400mm as the display distance
    if display_distance_mm is None or display_distance_mm <= 0:
        display_distance_mm = 400.0  # 40cm = 400mm
    return monitor_width, monitor_height

def calculate_screen_position_from_angle(angle_degrees, is_horizontal=True):
    """
    Calculate screen position (in pixels) from viewing angle in degrees.
    
    Args:
        angle_degrees: Viewing angle in degrees
        is_horizontal: True for horizontal movement, False for vertical
    
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
        if is_horizontal:
            screen_x = monitor_width // 2 + int(offset_pixels)
            screen_y = monitor_height // 2  # Keep at vertical center
        else:
            screen_x = monitor_width // 2  # Keep at horizontal center
            screen_y = monitor_height // 2 - int(offset_pixels)  # Invert Y for screen coords
        
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

def start_smooth_pursuit_test():
    """Start smooth pursuit testing"""
    global smooth_pursuit_active, showing_instruction, instruction_start_time
    global showing_target, target_start_time, phase, phase_start_time
    global pursuit_data, gaze_samples, target_frequency, target_amplitude_pixels
    global monitor_width, monitor_height
    
    if monitor_width is None or monitor_height is None:
        get_monitor_resolution()
    
    if monitor_width is None or monitor_height is None:
        print("Error: Could not get monitor resolution for smooth pursuit testing")
        return False
    
    # Initialize test
    pursuit_data = []
    gaze_samples = []
    smooth_pursuit_active = True
    showing_instruction = True
    showing_target = False
    instruction_start_time = time.time()
    target_start_time = None
    phase = 'horizontal'
    phase_start_time = None
    current_target_position = (monitor_width // 2, monitor_height // 2)
    
    # Randomly select frequency between 0.2Hz and 0.5Hz
    target_frequency = random.uniform(FREQUENCY_MIN, FREQUENCY_MAX)
    
    # Calculate amplitude in pixels from degrees
    # For ±15 degrees, calculate the pixel offset
    max_angle_rad = math.radians(target_amplitude_deg)
    max_offset_mm = math.tan(max_angle_rad) * display_distance_mm
    
    import eye_tracking
    if eye_tracking.display_width_mm is not None and eye_tracking.display_width_mm > 0:
        pixels_per_mm = monitor_width / eye_tracking.display_width_mm
    else:
        pixels_per_mm = 96.0 / 25.4
    
    target_amplitude_pixels = max_offset_mm * pixels_per_mm
    
    print("Smooth pursuit test started!")
    print(f"  Frequency: {target_frequency:.2f} Hz")
    print(f"  Amplitude: ±{target_amplitude_deg} degrees ({target_amplitude_pixels:.1f} pixels)")
    print("  Instruction will be shown first")
    print("  Then horizontal movement for 15 seconds")
    print("  Press 'p' again to stop early")
    
    return True

def stop_smooth_pursuit_test():
    """Stop smooth pursuit testing and calculate results"""
    global smooth_pursuit_active, pursuit_data
    
    if not smooth_pursuit_active:
        return None
    
    smooth_pursuit_active = False
    
    if len(pursuit_data) == 0:
        print("No smooth pursuit data collected.")
        return None
    
    try:
        # Calculate metrics
        results = calculate_pursuit_metrics(pursuit_data)
        
        # Save data (include gaze_samples for visualization)
        global gaze_samples
        save_pursuit_data(pursuit_data, results, gaze_samples)
        
        # Print results
        print_pursuit_results(results)
        
        return results
    except Exception as e:
        print(f"Error stopping smooth pursuit test: {e}")
        import traceback
        traceback.print_exc()
        return None

def update_smooth_pursuit_test(gaze_x, gaze_y):
    """Update smooth pursuit testing state - called from main loop"""
    global showing_instruction, instruction_start_time
    global showing_target, target_start_time, phase, phase_start_time
    global current_target_position, pursuit_data, gaze_samples
    global monitor_width, monitor_height, target_frequency, target_amplitude_pixels
    global display_distance_mm
    
    if not smooth_pursuit_active:
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
            # Keep only last 20 seconds of samples (enough for horizontal phase)
            cutoff_time = current_time_ms - 20000
            gaze_samples = [(t, x, y) for t, x, y in gaze_samples if t > cutoff_time]
        
        # State machine: instruction -> horizontal phase -> done
        
        if showing_instruction:
            elapsed_ms = (current_time - instruction_start_time) * 1000
            if elapsed_ms >= INSTRUCTION_DURATION_MS:
                # Move to showing target (horizontal phase)
                showing_instruction = False
                showing_target = True
                target_start_time = current_time
                phase_start_time = current_time
                phase = 'horizontal'
                current_target_position = (monitor_width // 2, monitor_height // 2)
        
        elif showing_target:
            elapsed = current_time - phase_start_time
            
            # Check if phase duration exceeded
            if elapsed >= PHASE_DURATION_SEC:
                # Horizontal phase complete - stop the test
                stop_smooth_pursuit_test()
                return
            
            # Calculate target position based on sinusoidal motion
            time_from_phase_start = current_time - phase_start_time
            
            # Sinusoidal motion: position = amplitude * sin(2 * pi * frequency * time)
            angle_offset_deg = target_amplitude_deg * math.sin(2 * math.pi * target_frequency * time_from_phase_start)
            
            # Only horizontal phase now
            target_x, target_y = calculate_screen_position_from_angle(angle_offset_deg, is_horizontal=True)
            
            current_target_position = (target_x, target_y)
            
            # Calculate target velocity (for gain calculation)
            # velocity = d(position)/dt = amplitude * 2 * pi * frequency * cos(2 * pi * frequency * time)
            target_velocity_deg_per_s = (target_amplitude_deg * 2 * math.pi * target_frequency * 
                                        math.cos(2 * math.pi * target_frequency * time_from_phase_start))
            
            # Convert to pixels per second
            max_angle_rad = math.radians(target_amplitude_deg)
            max_offset_mm = math.tan(max_angle_rad) * display_distance_mm
            
            import eye_tracking
            if eye_tracking.display_width_mm is not None and eye_tracking.display_width_mm > 0:
                pixels_per_mm = monitor_width / eye_tracking.display_width_mm
            else:
                pixels_per_mm = 96.0 / 25.4
            
            # Velocity in mm/s
            target_velocity_mm_per_s = (target_velocity_deg_per_s * math.pi / 180.0) * display_distance_mm
            target_velocity_pixels_per_s = target_velocity_mm_per_s * pixels_per_mm
            
            # Calculate eye velocity and gain
            if gaze_x is not None and gaze_y is not None and len(gaze_samples) >= 2:
                # Get recent gaze samples (last 200ms for velocity calculation)
                recent_samples = [(t, x, y) for t, x, y in gaze_samples 
                                if current_time_ms - t <= 200]
                
                if len(recent_samples) >= 2:
                    # Calculate eye velocity
                    eye_velocity = calculate_eye_velocity(recent_samples, phase)
                    
                    # Calculate gain (eye velocity / target velocity)
                    if abs(target_velocity_pixels_per_s) > 0.1:  # Avoid division by zero
                        gain = eye_velocity / target_velocity_pixels_per_s
                    else:
                        gain = 0.0
                    
                    # Calculate latency (only for first few seconds of each phase)
                    latency_ms = None
                    if time_from_phase_start < 2.0:  # Only calculate latency in first 2 seconds
                        latency_ms = detect_pursuit_initiation(phase_start_time, phase, target_frequency)
                    
                    # Record measurement
                    measurement = {
                        'timestamp': current_time,
                        'phase': phase,
                        'time_from_phase_start': time_from_phase_start,
                        'target_x': target_x,
                        'target_y': target_y,
                        'target_velocity_pixels_per_s': target_velocity_pixels_per_s,
                        'target_velocity_deg_per_s': target_velocity_deg_per_s,
                        'eye_velocity_pixels_per_s': eye_velocity,
                        'gain': gain,
                        'latency_ms': latency_ms
                    }
                    pursuit_data.append(measurement)
    except Exception as e:
        print(f"Error in update_smooth_pursuit_test: {e}")
        import traceback
        traceback.print_exc()
        # Don't stop the test, just log the error

def calculate_eye_velocity(recent_samples, phase):
    """
    Calculate eye velocity from recent gaze samples.
    
    Args:
        recent_samples: List of (timestamp_ms, x, y) tuples
        phase: 'horizontal' or 'vertical'
    
    Returns:
        Eye velocity in pixels per second
    """
    if len(recent_samples) < 2:
        return 0.0
    
    # Sort by timestamp
    sorted_samples = sorted(recent_samples, key=lambda s: s[0])
    
    # Calculate velocity from first to last sample
    t1, x1, y1 = sorted_samples[0]
    t2, x2, y2 = sorted_samples[-1]
    
    dt_ms = t2 - t1
    if dt_ms <= 0:
        return 0.0
    
    if phase == 'horizontal':
        # Horizontal velocity
        dx = x2 - x1
        velocity_pixels_per_s = (dx / dt_ms) * 1000.0
    else:
        # Vertical velocity
        dy = y2 - y1
        velocity_pixels_per_s = (dy / dt_ms) * 1000.0
    
    return velocity_pixels_per_s

def detect_pursuit_initiation(phase_start_time, phase, frequency):
    """
    Detect when smooth pursuit is initiated (eye starts following target).
    
    Args:
        phase_start_time: Time when phase started
        phase: 'horizontal' or 'vertical'
        frequency: Target frequency
    
    Returns:
        Latency in ms, or None if not detected
    """
    global gaze_samples, monitor_width, monitor_height
    
    if monitor_width is None:
        get_monitor_resolution()
    
    current_time = time.time()
    current_time_ms = current_time * 1000
    phase_start_ms = phase_start_time * 1000
    
    # Look at samples in first 2 seconds after phase start
    window_start = phase_start_ms
    window_end = phase_start_ms + 2000
    
    relevant_samples = [(t, x, y) for t, x, y in gaze_samples 
                       if window_start <= t <= window_end]
    
    if len(relevant_samples) < 3:
        return None
    
    # Calculate expected target velocity at phase start
    # At t=0, velocity is maximum: v = amplitude * 2 * pi * frequency
    global target_amplitude_deg, display_distance_mm
    max_velocity_deg_per_s = target_amplitude_deg * 2 * math.pi * frequency
    
    # Convert to pixels per second
    max_angle_rad = math.radians(target_amplitude_deg)
    max_offset_mm = math.tan(max_angle_rad) * display_distance_mm
    
    import eye_tracking
    if eye_tracking.display_width_mm is not None and eye_tracking.display_width_mm > 0:
        pixels_per_mm = monitor_width / eye_tracking.display_width_mm
    else:
        pixels_per_mm = 96.0 / 25.4
    
    max_velocity_mm_per_s = (max_velocity_deg_per_s * math.pi / 180.0) * display_distance_mm
    expected_velocity_pixels_per_s = max_velocity_mm_per_s * pixels_per_mm
    
    # For vertical movement, Y axis is inverted (up = negative Y)
    if phase == 'vertical':
        expected_velocity_pixels_per_s = -expected_velocity_pixels_per_s
    
    # Look for eye movement that matches target direction and has reasonable velocity
    # We'll use a sliding window to detect when eye velocity becomes significant
    window_size = 5  # Use 5 samples for velocity calculation
    
    for i in range(len(relevant_samples) - window_size):
        window_samples = relevant_samples[i:i+window_size]
        eye_velocity = calculate_eye_velocity(window_samples, phase)
        
        # Check if eye velocity is in same direction as target and exceeds threshold
        # Threshold: at least 30% of expected velocity
        threshold = abs(expected_velocity_pixels_per_s) * 0.3
        
        if abs(eye_velocity) > threshold:
            # Check direction (should match target direction at phase start)
            # At phase start, target moves in positive X direction (horizontal) or negative Y direction (vertical, up)
            if phase == 'horizontal':
                # Horizontal: positive velocity means moving right
                if eye_velocity > 0:
                    # Pursuit initiated
                    latency_ms = window_samples[0][0] - phase_start_ms
                    return max(0, latency_ms)
            else:
                # Vertical: negative velocity means moving up (Y decreases)
                if eye_velocity < 0:
                    # Pursuit initiated
                    latency_ms = window_samples[0][0] - phase_start_ms
                    return max(0, latency_ms)
    
    return None

def calculate_pursuit_metrics(data):
    """Calculate average metrics from collected pursuit data"""
    if len(data) == 0:
        return None
    
    # Separate by phase
    horizontal_data = [d for d in data if d.get('phase') == 'horizontal']
    vertical_data = [d for d in data if d.get('phase') == 'vertical']
    
    # Calculate gains (exclude outliers)
    gains = [d['gain'] for d in data if d.get('gain') is not None and -2.0 <= d['gain'] <= 2.0]
    horizontal_gains = [d['gain'] for d in horizontal_data if d.get('gain') is not None and -2.0 <= d['gain'] <= 2.0]
    vertical_gains = [d['gain'] for d in vertical_data if d.get('gain') is not None and -2.0 <= d['gain'] <= 2.0]
    
    # Calculate latencies (only valid ones)
    latencies = [d['latency_ms'] for d in data if d.get('latency_ms') is not None and d['latency_ms'] >= 0]
    horizontal_latencies = [d['latency_ms'] for d in horizontal_data if d.get('latency_ms') is not None and d['latency_ms'] >= 0]
    vertical_latencies = [d['latency_ms'] for d in vertical_data if d.get('latency_ms') is not None and d['latency_ms'] >= 0]
    
    results = {
        'total_measurements': len(data),
        'horizontal_measurements': len(horizontal_data),
        'vertical_measurements': len(vertical_data),
        'average_gain': np.mean(gains) if gains else 0.0,
        'average_gain_horizontal': np.mean(horizontal_gains) if horizontal_gains else 0.0,
        'average_gain_vertical': np.mean(vertical_gains) if vertical_gains else 0.0,
        'std_gain': np.std(gains) if gains else 0.0,
        'average_latency_ms': np.mean(latencies) if latencies else 0.0,
        'average_latency_horizontal_ms': np.mean(horizontal_latencies) if horizontal_latencies else 0.0,
        'average_latency_vertical_ms': np.mean(vertical_latencies) if vertical_latencies else 0.0,
        'std_latency_ms': np.std(latencies) if latencies else 0.0
    }
    
    return results

def save_pursuit_data(data, results, gaze_samples=None):
    """Save smooth pursuit data to JSON file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    json_filename = f"smooth_pursuit_results_{timestamp}.json"
    try:
        global monitor_width, monitor_height, target_frequency, target_amplitude_deg
        json_data = {
            'metadata': {
                'timestamp': timestamp,
                'test_date': datetime.now().isoformat(),
                'monitor_width': monitor_width,
                'monitor_height': monitor_height,
                'target_frequency_hz': target_frequency,
                'target_amplitude_degrees': target_amplitude_deg,
                'phase_duration_seconds': PHASE_DURATION_SEC
            },
            'results': results,
            'raw_data': data
        }
        
        # Add gaze path data if available (convert tuples to lists for JSON serialization)
        if gaze_samples:
            json_data['gaze_path'] = [[t, x, y] for t, x, y in gaze_samples]
        with open(json_filename, 'w') as jsonfile:
            json.dump(json_data, jsonfile, indent=2)
        print(f"✓ Smooth pursuit results saved to {json_filename}")
    except Exception as e:
        print(f"✗ Error saving JSON: {e}")
        import traceback
        traceback.print_exc()

def print_pursuit_results(results):
    """Print smooth pursuit results to console"""
    if results is None:
        return
    
    print("\n" + "="*60)
    print("SMOOTH PURSUIT TEST RESULTS")
    print("="*60)
    print(f"Total Measurements: {results['total_measurements']}")
    print(f"  Horizontal: {results['horizontal_measurements']}")
    print(f"  Vertical: {results['vertical_measurements']}")
    print(f"\nAverage Gain (eye velocity / target velocity):")
    print(f"  Overall:        {results['average_gain']:.3f} (std: {results['std_gain']:.3f})")
    print(f"  Horizontal:     {results['average_gain_horizontal']:.3f}")
    print(f"  Vertical:       {results['average_gain_vertical']:.3f}")
    print(f"\nAverage Latency:")
    if results['average_latency_ms'] > 0:
        print(f"  Overall:        {results['average_latency_ms']:.2f} ms (std: {results['std_latency_ms']:.2f})")
        print(f"  Horizontal:     {results['average_latency_horizontal_ms']:.2f} ms")
        print(f"  Vertical:       {results['average_latency_vertical_ms']:.2f} ms")
    else:
        print("  No latency data collected")
    print(f"{'='*60}\n")

def get_pursuit_overlay_draw_function():
    """Returns a function that draws smooth pursuit testing elements on the overlay canvas"""
    def draw_on_overlay(canvas):
        global smooth_pursuit_active, showing_instruction, showing_target
        global current_target_position, monitor_width, monitor_height
        
        try:
            if not smooth_pursuit_active:
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
                    center_x, center_y - 50,
                    text="Follow the dot as smoothly as possible with your eyes.",
                    fill='white', font=('Arial', 24, 'bold'),
                    justify='center'
                )
                canvas.create_text(
                    center_x, center_y + 10,
                    text="Do not let your eyes jump ahead.",
                    fill='white', font=('Arial', 24, 'bold'),
                    justify='center'
                )
            
            # Show moving target
            elif showing_target:
                if current_target_position is not None:
                    try:
                        x, y = current_target_position
                        # Validate coordinates are within screen bounds
                        if x < 0 or x > monitor_width or y < 0 or y > monitor_height:
                            # Skip drawing if coordinates are invalid
                            return
                        
                        # Draw target circle
                        dot_radius = 10
                        color = 'cyan'  # Cyan for smooth pursuit target
                        canvas.create_oval(
                            x - dot_radius, y - dot_radius,
                            x + dot_radius, y + dot_radius,
                            fill=color, outline=color, width=2
                        )
                    except (TypeError, ValueError) as e:
                        # Handle invalid coordinates
                        print(f"Warning: Invalid target position: {current_target_position}")
                        pass
        except Exception as e:
            # Print error for debugging but don't crash
            print(f"Error in smooth pursuit overlay drawing: {e}")
            import traceback
            traceback.print_exc()
    
    return draw_on_overlay

