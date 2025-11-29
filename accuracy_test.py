import numpy as np
import math
import time
import csv
import json
from datetime import datetime

# Accuracy testing globals
accuracy_testing_active = False
accuracy_data = []  # List to store accuracy measurements
target_position = None  # Current target position (x, y)
target_start_time = None
target_duration = 2.0  # How long to stay at each target (seconds)
target_path = []  # List of target positions to visit
target_index = 0
accuracy_test_start_time = None
monitor_width = None
monitor_height = None

def get_monitor_resolution():
    """Get primary monitor resolution - import from eye_tracking module"""
    import eye_tracking
    global monitor_width, monitor_height
    monitor_width, monitor_height = eye_tracking.monitor_width, eye_tracking.monitor_height
    return monitor_width, monitor_height

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

def start_accuracy_testing(grid_size=3):
    """Start accuracy testing with moving target"""
    global accuracy_testing_active, target_path, target_index, target_position
    global target_start_time, accuracy_data, accuracy_test_start_time, monitor_width, monitor_height
    
    if monitor_width is None or monitor_height is None:
        get_monitor_resolution()
    
    if monitor_width is None or monitor_height is None:
        print("Error: Could not get monitor resolution for accuracy testing")
        return False
    
    # Generate target path
    target_path = generate_target_path(grid_size)
    if len(target_path) == 0:
        print("Error: Could not generate target path")
        return False
        
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
    
    return True

def stop_accuracy_testing():
    """Stop accuracy testing and calculate results"""
    global accuracy_testing_active, accuracy_data, target_path, target_position
    
    if not accuracy_testing_active:
        return None
    
    # Save current state before clearing
    was_active = accuracy_testing_active
    data_to_save = accuracy_data.copy() if accuracy_data else []
    path_to_save = target_path.copy() if target_path else []
    
    # Clear state first to prevent overlay from trying to draw
    accuracy_testing_active = False
    target_position = None
    
    try:
        if len(data_to_save) == 0:
            print("No accuracy data collected.")
            return None
        
        # Calculate accuracy metrics
        results = calculate_accuracy_metrics(data_to_save)
        
        if results is None:
            print("Error calculating accuracy metrics.")
            return None
        
        # Save data (use saved copy)
        save_accuracy_data(data_to_save, results, path_to_save)
        
        # Print results
        print_accuracy_results(results)
        
        return results
    except Exception as e:
        print(f"Error stopping accuracy test: {e}")
        import traceback
        traceback.print_exc()
        return None

def update_accuracy_target():
    """Update target position based on time"""
    global target_position, target_index, target_path, target_start_time, target_duration, accuracy_testing_active
    
    try:
        if not accuracy_testing_active or len(target_path) == 0:
            return
        
        current_time = time.time()
        elapsed = current_time - target_start_time
        
        # Move to next target if duration exceeded
        if elapsed >= target_duration:
            target_index += 1
            if target_index >= len(target_path):
                # All targets visited, stop testing
                # Use a flag to avoid re-entering if already stopping
                if accuracy_testing_active:
                    stop_accuracy_testing()
                return
            target_position = target_path[target_index]
            target_start_time = current_time
    except Exception as e:
        print(f"Error updating accuracy target: {e}")
        import traceback
        traceback.print_exc()

def record_accuracy_measurement(gaze_x, gaze_y):
    """Record a single accuracy measurement"""
    global accuracy_data, target_position, accuracy_testing_active, target_index
    
    try:
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
    except Exception as e:
        # Silently handle errors to prevent crashes during measurement recording
        # Errors can occur if state changes during recording
        pass

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

def save_accuracy_data(data, results, target_path_copy=None):
    """Save accuracy data to CSV and JSON files"""
    global target_path, monitor_width, monitor_height
    
    # Use provided copy or fall back to global
    if target_path_copy is None:
        target_path_copy = target_path if target_path else []
    
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
                'target_count': len(target_path_copy) if target_path_copy else 0
            },
            'results': results,
            'raw_data': data
        }
        with open(json_filename, 'w') as jsonfile:
            json.dump(json_data, jsonfile, indent=2)
        print(f"✓ Accuracy results saved to {json_filename}")
    except Exception as e:
        print(f"✗ Error saving JSON: {e}")
        import traceback
        traceback.print_exc()

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

def get_accuracy_overlay_draw_function():
    """Returns a function that draws accuracy testing elements on the overlay canvas"""
    def draw_on_overlay(canvas):
        global accuracy_testing_active, target_position, target_index, target_path
        global monitor_width, monitor_height
        
        try:
            # Early return if test is not active or no target
            if not accuracy_testing_active or target_position is None:
                return
            
            # Get monitor dimensions if not set
            if monitor_width is None or monitor_height is None:
                get_monitor_resolution()
            
            # Double-check target_position is still valid (thread safety)
            if target_position is None:
                return
            
            target_x, target_y = target_position
            # Draw large target circle (green)
            target_radius = 30
            canvas.create_oval(
                target_x - target_radius, target_y - target_radius,
                target_x + target_radius, target_y + target_radius,
                fill='green', outline='darkgreen', width=3
            )
            # Draw inner white dot
            canvas.create_oval(
                target_x - 8, target_y - 8,
                target_x + 8, target_y + 8,
                fill='white', outline='white'
            )
            # Show target number
            if target_path:
                canvas.create_text(
                    target_x, target_y - target_radius - 20,
                    text=f"Target {target_index + 1}/{len(target_path)}",
                    fill='green', font=('Arial', 14, 'bold')
                )
            
            # Draw line from target to gaze if gaze position is available
            import eye_tracking
            if eye_tracking.current_gaze_x is not None and eye_tracking.current_gaze_y is not None:
                gaze_x = eye_tracking.current_gaze_x
                gaze_y = eye_tracking.current_gaze_y
                canvas.create_line(
                    target_x, target_y, gaze_x, gaze_y,
                    fill='yellow', width=2, dash=(5, 5)
                )
        except Exception as e:
            # Silently handle errors in overlay drawing to prevent crashes
            # Errors can occur if state changes during drawing
            pass
    
    return draw_on_overlay

def update_accuracy_test(gaze_x, gaze_y):
    """Update accuracy testing state - called from main loop"""
    global accuracy_testing_active
    
    if accuracy_testing_active:
        update_accuracy_target()
        record_accuracy_measurement(gaze_x, gaze_y)

