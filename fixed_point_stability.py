import numpy as np
import math
import time
import json
from datetime import datetime

# Fixed point stability testing globals
fixed_point_active = False
stability_data = []  # List to store stability measurements
current_target_position = None  # Target position (center of screen)
test_start_time = None
showing_instruction = False
instruction_start_time = None
showing_target = False
target_start_time = None

# Timing parameters
INSTRUCTION_DURATION_MS = 3000  # How long to show instruction (ms)
TEST_DURATION_SEC = 20.0  # Duration of test in seconds

# Store gaze samples for stability tracking
gaze_samples = []  # List of (timestamp, x, y, deviation_deg) tuples
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

def calculate_deviation_from_center(gaze_x, gaze_y, center_x, center_y):
    """
    Calculate angular deviation from center in degrees.
    
    Args:
        gaze_x, gaze_y: Current gaze position in pixels
        center_x, center_y: Center target position in pixels
    
    Returns:
        Deviation angle in degrees
    """
    global display_distance_mm, monitor_width
    
    try:
        if monitor_width is None or monitor_height is None:
            get_monitor_resolution()
        
        if display_distance_mm is None or display_distance_mm <= 0:
            display_distance_mm = 400.0  # 40cm = 400mm
        
        # Calculate pixel offset from center
        dx_pixels = gaze_x - center_x
        dy_pixels = gaze_y - center_y
        
        # Calculate distance in pixels
        distance_pixels = math.sqrt(dx_pixels**2 + dy_pixels**2)
        
        # Convert pixels to mm
        import eye_tracking
        if eye_tracking.display_width_mm is not None and eye_tracking.display_width_mm > 0:
            pixels_per_mm = monitor_width / eye_tracking.display_width_mm
        else:
            # Fallback: assume 96 DPI
            pixels_per_mm = 96.0 / 25.4  # pixels per mm at 96 DPI
        
        distance_mm = distance_pixels / pixels_per_mm
        
        # Calculate angular deviation
        # tan(angle) = distance / display_distance
        if distance_mm > 0 and display_distance_mm > 0:
            angle_rad = math.atan2(distance_mm, display_distance_mm)
            angle_deg = math.degrees(angle_rad)
        else:
            angle_deg = 0.0
        
        return angle_deg
    except Exception as e:
        print(f"Error calculating deviation: {e}")
        import traceback
        traceback.print_exc()
        return 0.0

def start_fixed_point_test():
    """Start fixed point stability testing"""
    global fixed_point_active, showing_instruction, instruction_start_time
    global showing_target, target_start_time, stability_data, gaze_samples
    global current_target_position, monitor_width, monitor_height
    
    if monitor_width is None or monitor_height is None:
        get_monitor_resolution()
    
    if monitor_width is None or monitor_height is None:
        print("Error: Could not get monitor resolution for fixed point stability testing")
        return False
    
    # Initialize test
    stability_data = []
    gaze_samples = []
    fixed_point_active = True
    showing_instruction = True
    showing_target = False
    instruction_start_time = time.time()
    target_start_time = None
    current_target_position = (monitor_width // 2, monitor_height // 2)
    
    print("Fixed point stability test started!")
    print("  Instruction will be shown first")
    print("  Then a dot will appear in the center for 20 seconds")
    print("  Stare at the dot without moving your eyes")
    print("  Press 'f' again to stop early")
    
    return True

def stop_fixed_point_test():
    """Stop fixed point stability testing and calculate results"""
    global fixed_point_active, stability_data
    
    if not fixed_point_active:
        return None
    
    fixed_point_active = False
    
    if len(stability_data) == 0:
        print("No stability data collected.")
        return None
    
    try:
        # Calculate metrics
        results = calculate_stability_metrics(stability_data)
        
        # Save data
        save_stability_data(stability_data, results)
        
        # Print results
        print_stability_results(results)
        
        return results
    except Exception as e:
        print(f"Error stopping fixed point stability test: {e}")
        import traceback
        traceback.print_exc()
        return None

def update_fixed_point_test(gaze_x, gaze_y):
    """Update fixed point stability testing state - called from main loop"""
    global showing_instruction, instruction_start_time
    global showing_target, target_start_time, stability_data
    global current_target_position, monitor_width, monitor_height
    
    if not fixed_point_active:
        return
    
    try:
        # Ensure monitor resolution is available
        if monitor_width is None or monitor_height is None:
            get_monitor_resolution()
        
        current_time = time.time()
        current_time_ms = current_time * 1000
        
        # State machine: instruction -> target -> done
        
        if showing_instruction:
            elapsed_ms = (current_time - instruction_start_time) * 1000
            if elapsed_ms >= INSTRUCTION_DURATION_MS:
                # Move to showing target
                showing_instruction = False
                showing_target = True
                target_start_time = current_time
                current_target_position = (monitor_width // 2, monitor_height // 2)
        
        elif showing_target:
            elapsed = current_time - target_start_time
            
            # Check if test duration exceeded
            if elapsed >= TEST_DURATION_SEC:
                stop_fixed_point_test()
                return
            
            # Record gaze deviation if gaze data is available
            if gaze_x is not None and gaze_y is not None:
                center_x, center_y = current_target_position
                deviation_deg = calculate_deviation_from_center(gaze_x, gaze_y, center_x, center_y)
                
                # Record measurement
                measurement = {
                    'timestamp': current_time,
                    'time_from_start': elapsed,
                    'gaze_x': gaze_x,
                    'gaze_y': gaze_y,
                    'target_x': center_x,
                    'target_y': center_y,
                    'deviation_degrees': deviation_deg
                }
                stability_data.append(measurement)
                gaze_samples.append((current_time_ms, gaze_x, gaze_y, deviation_deg))
    except Exception as e:
        print(f"Error in update_fixed_point_test: {e}")
        import traceback
        traceback.print_exc()
        # Don't stop the test, just log the error

def calculate_stability_metrics(data):
    """Calculate average metrics from collected stability data"""
    if len(data) == 0:
        return None
    
    deviations = [d['deviation_degrees'] for d in data if d.get('deviation_degrees') is not None]
    
    if len(deviations) == 0:
        return None
    
    results = {
        'total_measurements': len(data),
        'average_deviation_degrees': np.mean(deviations),
        'median_deviation_degrees': np.median(deviations),
        'std_deviation_degrees': np.std(deviations),
        'max_deviation_degrees': np.max(deviations),
        'min_deviation_degrees': np.min(deviations),
        'rms_deviation_degrees': np.sqrt(np.mean([d**2 for d in deviations]))
    }
    
    return results

def save_stability_data(data, results):
    """Save fixed point stability data to JSON file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    json_filename = f"fixed_point_stability_results_{timestamp}.json"
    try:
        global monitor_width, monitor_height
        json_data = {
            'metadata': {
                'timestamp': timestamp,
                'test_date': datetime.now().isoformat(),
                'monitor_width': monitor_width,
                'monitor_height': monitor_height,
                'test_duration_seconds': TEST_DURATION_SEC
            },
            'results': results,
            'raw_data': data
        }
        with open(json_filename, 'w') as jsonfile:
            json.dump(json_data, jsonfile, indent=2)
        print(f"✓ Fixed point stability results saved to {json_filename}")
    except Exception as e:
        print(f"✗ Error saving JSON: {e}")
        import traceback
        traceback.print_exc()

def print_stability_results(results):
    """Print fixed point stability results to console"""
    if results is None:
        return
    
    print("\n" + "="*60)
    print("FIXED POINT STABILITY TEST RESULTS")
    print("="*60)
    print(f"Total Measurements: {results['total_measurements']}")
    print(f"\nDeviation Statistics (degrees):")
    print(f"  Average Deviation: {results['average_deviation_degrees']:.4f}°")
    print(f"  Median Deviation:  {results['median_deviation_degrees']:.4f}°")
    print(f"  RMS Deviation:     {results['rms_deviation_degrees']:.4f}°")
    print(f"  Std Deviation:     {results['std_deviation_degrees']:.4f}°")
    print(f"  Max Deviation:     {results['max_deviation_degrees']:.4f}°")
    print(f"  Min Deviation:     {results['min_deviation_degrees']:.4f}°")
    print(f"{'='*60}\n")

def get_stability_overlay_draw_function():
    """Returns a function that draws fixed point stability testing elements on the overlay canvas"""
    def draw_on_overlay(canvas):
        global fixed_point_active, showing_instruction, showing_target
        global current_target_position, monitor_width, monitor_height
        
        try:
            if not fixed_point_active:
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
                    center_x, center_y - 20,
                    text="Stare without moving your eyes",
                    fill='white', font=('Arial', 24, 'bold'),
                    justify='center'
                )
                canvas.create_text(
                    center_x, center_y + 30,
                    text="for the full duration of the test.",
                    fill='white', font=('Arial', 24, 'bold'),
                    justify='center'
                )
            
            # Show target dot
            elif showing_target:
                if current_target_position is not None:
                    try:
                        x, y = current_target_position
                        # Validate coordinates are within screen bounds
                        if x < 0 or x > monitor_width or y < 0 or y > monitor_height:
                            # Skip drawing if coordinates are invalid
                            return
                        
                        # Draw target circle (small dot in center)
                        dot_radius = 6
                        color = 'yellow'  # Yellow for fixed point target
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
            print(f"Error in fixed point stability overlay drawing: {e}")
            import traceback
            traceback.print_exc()
    
    return draw_on_overlay

