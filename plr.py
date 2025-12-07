import numpy as np
import time
import json
from datetime import datetime

# PLR test globals
plr_test_active = False
plr_baseline_samples = []  # Pupil diameter samples before light flash (baseline)
plr_response_samples = []  # Pupil diameter samples during/after light flash
plr_flash_start_time = None
plr_baseline_start_time = None
plr_baseline_duration_sec = 1.0  # Collect baseline for 1 second before flash
showing_plr_flash = False
plr_current_result = None  # Store latest PLR metrics for dashboard access

# Timing parameters
PLR_FLASH_DURATION_SEC = 2.5  # Duration of white flash for PLR test

monitor_width = None
monitor_height = None

def get_monitor_resolution():
    """Get primary monitor resolution - import from eye_tracking module"""
    import eye_tracking
    global monitor_width, monitor_height
    monitor_width, monitor_height = eye_tracking.monitor_width, eye_tracking.monitor_height
    return monitor_width, monitor_height

def get_plr_overlay_draw_function():
    """Returns a function that draws PLR flash on the overlay canvas"""
    def draw_on_overlay(canvas):
        global plr_test_active, showing_plr_flash, monitor_width, monitor_height
        
        try:
            if not plr_test_active or not showing_plr_flash:
                return
            
            # Ensure monitor resolution is available
            if monitor_width is None or monitor_height is None:
                get_monitor_resolution()
            
            if monitor_width is None or monitor_height is None:
                # Can't draw without monitor resolution
                return
            
            # Draw fullscreen white flash
            canvas.create_rectangle(
                0, 0, monitor_width, monitor_height,
                fill='white', outline='white', width=0
            )
        except Exception as e:
            # Silently handle errors
            pass
    
    return draw_on_overlay

def start_plr_test():
    """Start PLR (Pupillary Light Reflex) test"""
    global plr_test_active, plr_baseline_samples, plr_response_samples
    global plr_flash_start_time, plr_baseline_start_time, showing_plr_flash
    global monitor_width, monitor_height, plr_current_result
    
    if monitor_width is None or monitor_height is None:
        get_monitor_resolution()
    
    # Initialize PLR test
    plr_test_active = True
    plr_baseline_samples = []
    plr_response_samples = []
    plr_baseline_start_time = time.time()
    plr_flash_start_time = None
    showing_plr_flash = False
    plr_current_result = None  # Clear previous results
    
    print("PLR (Pupillary Light Reflex) test started!")
    print("  Collecting baseline pupil diameter (1 second)...")
    print("  Then screen will flash white for 2.5 seconds")
    print("  Keep your eyes open and look at the screen")
    print("  Press 'l' again to stop early")

def stop_plr_test():
    """Stop PLR test and calculate results"""
    global plr_test_active, plr_baseline_samples, plr_response_samples
    global showing_plr_flash, plr_current_result
    
    if not plr_test_active:
        return None
    
    plr_test_active = False
    showing_plr_flash = False
    
    if len(plr_baseline_samples) == 0 or len(plr_response_samples) == 0:
        print("Insufficient PLR data collected.")
        return None
    
    try:
        # Calculate PLR metrics
        result = calculate_plr_metrics(plr_baseline_samples, plr_response_samples)
        
        # Store result globally for dashboard access
        plr_current_result = result
        
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

