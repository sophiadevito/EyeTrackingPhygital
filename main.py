#!/usr/bin/env python3
"""
Main entry point for Eye Tracking Phygital Application

This script coordinates the eye tracking and accuracy testing modules.
Always uses real-time camera input (no video file selection).
"""

import argparse
import eye_tracking
import accuracy_test
import saccade_test
import smooth_pursuit
import fixed_point_stability
import calibration
import plr
import glob
import os
import report_generator

def main():
    """Main function that coordinates eye tracking and accuracy testing"""
    
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
    
    # Set up overlay callbacks for testing modules
    accuracy_overlay_func = accuracy_test.get_accuracy_overlay_draw_function()
    eye_tracking.add_overlay_callback(accuracy_overlay_func)
    
    saccade_overlay_func = saccade_test.get_saccade_overlay_draw_function()
    eye_tracking.add_overlay_callback(saccade_overlay_func)
    
    pursuit_overlay_func = smooth_pursuit.get_pursuit_overlay_draw_function()
    eye_tracking.add_overlay_callback(pursuit_overlay_func)
    
    stability_overlay_func = fixed_point_stability.get_stability_overlay_draw_function()
    eye_tracking.add_overlay_callback(stability_overlay_func)
    
    calibration_overlay_func = calibration.get_calibration_overlay_draw_function()
    eye_tracking.add_overlay_callback(calibration_overlay_func)
    
    # Add PLR overlay callback
    plr_overlay_func = plr.get_plr_overlay_draw_function()
    eye_tracking.add_overlay_callback(plr_overlay_func)
    
    # Add calibration overlay callback for automated suite
    calibration_suite_overlay_func = get_calibration_overlay_draw_function_for_suite()
    eye_tracking.add_overlay_callback(calibration_suite_overlay_func)
    
    # Define callback function for accuracy testing updates
    def accuracy_test_update(gaze_x, gaze_y):
        """Callback to update accuracy testing with gaze coordinates"""
        accuracy_test.update_accuracy_test(gaze_x, gaze_y)
    
    # Define callback function for saccade testing updates
    def saccade_test_update(gaze_x, gaze_y):
        """Callback to update saccade testing with gaze coordinates"""
        saccade_test.update_saccade_test(gaze_x, gaze_y)
    
    # Define callback function for smooth pursuit testing updates
    def pursuit_test_update(gaze_x, gaze_y):
        """Callback to update smooth pursuit testing with gaze coordinates"""
        smooth_pursuit.update_smooth_pursuit_test(gaze_x, gaze_y)
    
    # Define callback function for fixed point stability testing updates
    def stability_test_update(gaze_x, gaze_y):
        """Callback to update fixed point stability testing with gaze coordinates"""
        fixed_point_stability.update_fixed_point_test(gaze_x, gaze_y)
    
    # Define callback function for calibration updates
    def calibration_update(gaze_x, gaze_y, raw_gaze_x, raw_gaze_y, pupil_x, pupil_y):
        """Callback to update calibration with gaze coordinates"""
        calibration.update_calibration(gaze_x, gaze_y, raw_gaze_x, raw_gaze_y, pupil_x, pupil_y)
    
    # Define callback function for PLR updates
    def plr_update(pupil_diameter):
        """Callback to update PLR test with pupil diameter"""
        plr.update_plr_test(pupil_diameter)
    
    print("Starting Eye Tracking with Camera...")
    print("Controls:")
    print("  SPACEBAR - Pause/resume")
    print("  D        - Toggle debug mode")
    print("  C        - Calibrate (look at center and press 'c')")
    print("  G        - Toggle gaze overlay")
    print("  A        - Start/stop accuracy testing")
    print("  S        - Start/stop saccade testing")
    print("  P        - Start/stop smooth pursuit testing")
    print("  F        - Start/stop fixed point stability testing")
    print("  X        - Start 9-point calibration")
    print("  L        - Start/stop PLR (Pupillary Light Reflex) test")
    print("  Z        - Run automated test suite (Saccade → Pursuit → Fixed Point → PLR)")
    print("  B        - Show blink statistics")
    print("  Q        - Quit")
    print("")
    
    # We need to modify the eye tracking to handle test keys
    # Let's create a wrapper that handles this
    run_with_tests(args.debug, accuracy_test_update, saccade_test_update, pursuit_test_update, stability_test_update, calibration_update, plr_update)

# Module-level variable for calibration overlay state
automated_suite_calibration_overlay_active = False

def get_calibration_overlay_draw_function_for_suite():
    """Returns a function that draws calibration elements for automated suite"""
    def draw_on_overlay(canvas):
        global automated_suite_calibration_overlay_active
        import eye_tracking as et
        try:
            if not automated_suite_calibration_overlay_active:
                return
            
            # Ensure monitor resolution is available
            if et.monitor_width is None or et.monitor_height is None:
                et.get_monitor_resolution()
            
            if et.monitor_width is None or et.monitor_height is None:
                return
            
            center_x = et.monitor_width // 2
            center_y = et.monitor_height // 2
            
            # Draw instruction text
            canvas.create_text(
                center_x, center_y - 50,
                text="Look at the center of the screen",
                fill='white', font=('Arial', 24, 'bold'),
                justify='center'
            )
            
            # Draw white dot at center
            dot_radius = 10
            canvas.create_oval(
                center_x - dot_radius, center_y - dot_radius,
                center_x + dot_radius, center_y + dot_radius,
                fill='white', outline='white'
            )
        except Exception as e:
            pass
    
    return draw_on_overlay

def get_latest_test_results(test_type):
    """
    Find and parse the most recent JSON results file for a given test type.
    
    Args:
        test_type: One of 'saccade', 'smooth_pursuit', 'fixed_point', 'plr'
    
    Returns:
        Dictionary with results, or None if no file found
    """
    import json
    from datetime import datetime
    
    # Map test types to file patterns
    patterns = {
        'saccade': 'saccade_results_*.json',
        'smooth_pursuit': 'smooth_pursuit_results_*.json',
        'fixed_point': 'fixed_point_stability_results_*.json',
        'plr': 'plr_results_*.json'
    }
    
    if test_type not in patterns:
        return None
    
    # Find all matching files
    pattern = patterns[test_type]
    files = glob.glob(pattern)
    
    if not files:
        return None
    
    # Sort by modification time (newest first)
    files.sort(key=os.path.getmtime, reverse=True)
    latest_file = files[0]
    
    try:
        with open(latest_file, 'r') as f:
            data = json.load(f)
            # Extract results from the JSON structure
            if 'results' in data:
                return data['results']
    except Exception as e:
        print(f"Error reading {latest_file}: {e}")
        return None
    
    return None

def run_with_tests(debug_calibration_flag, accuracy_update_callback, saccade_update_callback, pursuit_update_callback, stability_update_callback, calibration_update_callback, plr_update_callback):
    """
    Run eye tracking with all testing modules integration
    Handles keyboard input for all tests
    """
    import cv2
    import eye_tracking as et
    global automated_suite_calibration_overlay_active  # Declare global for module-level variable
    
    # Set debug calibration flag
    et.debug_calibration = debug_calibration_flag
    
    # Always use camera input
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)  # Camera input for macOS
    cap.set(cv2.CAP_PROP_EXPOSURE, -5)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    # Initialize display dimensions and start overlay
    et.calculate_display_dimensions()
    et.start_gaze_overlay()
    
    debug_mode_on = False
    
    # Automated test suite state
    automated_suite_running = False
    automated_suite_state = None  # Will track which test we're on
    automated_suite_results = {}
    automated_suite_test_start_time = None
    automated_suite_initial_blinks = 0  # Store initial blink count when suite starts
    automated_suite_calibration_active = False
    automated_suite_calibration_start_time = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip camera horizontally to fix mirroring (if enabled)
        if et.FLIP_CAMERA_HORIZONTAL:
            frame = cv2.flip(frame, 1)  # 1 = horizontal flip

        # Crop and resize frame
        frame = et.crop_to_aspect_ratio(frame)

        #find the darkest point
        darkest_point = et.get_darkest_area(frame)

        if debug_mode_on:
            darkest_image = frame.copy()
            cv2.circle(darkest_image, darkest_point, 10, (0, 0, 255), -1)
            cv2.imshow('Darkest image patch', darkest_image)

        # Convert to grayscale to handle pixel value operations
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        darkest_pixel_value = gray_frame[darkest_point[1], darkest_point[0]]
        
        # apply thresholding operations at different levels
        # at least one should give us a good ellipse segment
        thresholded_image_strict = et.apply_binary_threshold(gray_frame, darkest_pixel_value, 5)#lite
        thresholded_image_strict = et.mask_outside_square(thresholded_image_strict, darkest_point, 250)

        thresholded_image_medium = et.apply_binary_threshold(gray_frame, darkest_pixel_value, 15)#medium
        thresholded_image_medium = et.mask_outside_square(thresholded_image_medium, darkest_point, 250)
        
        thresholded_image_relaxed = et.apply_binary_threshold(gray_frame, darkest_pixel_value, 25)#heavy
        thresholded_image_relaxed = et.mask_outside_square(thresholded_image_relaxed, darkest_point, 250)
        
        #take the three images thresholded at different levels and process them
        pupil_rotated_rect, pupil_center = et.process_frames(
            thresholded_image_strict, thresholded_image_medium, thresholded_image_relaxed, 
            frame, gray_frame, darkest_point, debug_mode_on, True, debug_calibration_flag
        )
        
        # Extract pupil position
        pupil_x, pupil_y = None, None
        if pupil_center is not None:
            pupil_x, pupil_y = pupil_center
        
        # Update accuracy testing if active
        if accuracy_test.accuracy_testing_active:
            accuracy_update_callback(et.current_gaze_x, et.current_gaze_y)
        
        # Update saccade testing if active
        if saccade_test.saccade_testing_active:
            try:
                saccade_update_callback(et.current_gaze_x, et.current_gaze_y)
            except Exception as e:
                print(f"Error updating saccade test: {e}")
                import traceback
                traceback.print_exc()
                # Don't stop the main loop, just log the error
        
        # Update smooth pursuit testing if active
        if smooth_pursuit.smooth_pursuit_active:
            try:
                pursuit_update_callback(et.current_gaze_x, et.current_gaze_y)
            except Exception as e:
                print(f"Error updating smooth pursuit test: {e}")
                import traceback
                traceback.print_exc()
                # Don't stop the main loop, just log the error
        
        # Update fixed point stability testing if active
        if fixed_point_stability.fixed_point_active:
            try:
                stability_update_callback(et.current_gaze_x, et.current_gaze_y)
            except Exception as e:
                print(f"Error updating fixed point stability test: {e}")
                import traceback
                traceback.print_exc()
                # Don't stop the main loop, just log the error
        
        # Update calibration if active
        if calibration.calibration_active:
            try:
                calibration_update_callback(
                    et.current_gaze_x, et.current_gaze_y,
                    et.raw_gaze_x, et.raw_gaze_y,
                    pupil_x, pupil_y
                )
            except Exception as e:
                print(f"Error updating calibration: {e}")
                import traceback
                traceback.print_exc()
                # Don't stop the main loop, just log the error
        
        # Update PLR test if active
        if plr.plr_test_active:
            try:
                # Update PLR test with current pupil diameter
                plr_update_callback(et.current_pupil_diameter)
            except Exception as e:
                print(f"Error updating PLR test: {e}")
                import traceback
                traceback.print_exc()
                # Don't stop the main loop, just log the error
        
        # Update gaze overlay if enabled - directly call update from video loop
        if et.overlay_running and et.current_gaze_x is not None and et.current_gaze_y is not None:
            # Direct update from video loop - this ensures updates happen
            try:
                # Call update directly instead of relying on periodic callback
                if et.gaze_overlay_window is not None and et.gaze_overlay_canvas is not None:
                    et.update_gaze_overlay(et.current_gaze_x, et.current_gaze_y)
                    # Also process tkinter events
                    et.gaze_overlay_window.update_idletasks()
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
            if et.raw_gaze_x is not None and et.raw_gaze_y is not None:
                et.calibrate_gaze(et.raw_gaze_x, et.raw_gaze_y)
            else:
                print("No gaze data available for calibration. Make sure pupil is detected.")
        
        # Toggle gaze overlay
        if key == ord('g'):
            if et.overlay_running:
                et.stop_gaze_overlay()
            else:
                et.start_gaze_overlay()
        
        # Start/stop accuracy testing
        if key == ord('a'):
            if not accuracy_test.accuracy_testing_active:
                if not et.overlay_running:
                    print("Please enable gaze overlay (press 'g') before starting accuracy testing.")
                else:
                    accuracy_test.start_accuracy_testing(grid_size=3)
            else:
                try:
                    accuracy_test.stop_accuracy_testing()
                except Exception as e:
                    print(f"Error stopping accuracy test: {e}")
                    import traceback
                    traceback.print_exc()
        
        # Start/stop saccade testing
        if key == ord('s'):
            if not saccade_test.saccade_testing_active:
                if not et.overlay_running:
                    print("Please enable gaze overlay (press 'g') before starting saccade testing.")
                else:
                    saccade_test.start_saccade_test()
            else:
                try:
                    saccade_test.stop_saccade_test()
                except Exception as e:
                    print(f"Error stopping saccade test: {e}")
                    import traceback
                    traceback.print_exc()
        
        # Start/stop smooth pursuit testing
        if key == ord('p'):
            if not smooth_pursuit.smooth_pursuit_active:
                if not et.overlay_running:
                    print("Please enable gaze overlay (press 'g') before starting smooth pursuit testing.")
                else:
                    smooth_pursuit.start_smooth_pursuit_test()
            else:
                try:
                    smooth_pursuit.stop_smooth_pursuit_test()
                except Exception as e:
                    print(f"Error stopping smooth pursuit test: {e}")
                    import traceback
                    traceback.print_exc()
        
        # Start/stop fixed point stability testing
        if key == ord('f'):
            if not fixed_point_stability.fixed_point_active:
                if not et.overlay_running:
                    print("Please enable gaze overlay (press 'g') before starting fixed point stability testing.")
                else:
                    fixed_point_stability.start_fixed_point_test()
            else:
                try:
                    fixed_point_stability.stop_fixed_point_test()
                except Exception as e:
                    print(f"Error stopping fixed point stability test: {e}")
                    import traceback
                    traceback.print_exc()
        
        # Start 9-point calibration
        if key == ord('x'):
            if not calibration.calibration_active:
                if not et.overlay_running:
                    print("Please enable gaze overlay (press 'g') before starting calibration.")
                else:
                    calibration.start_calibration()
            else:
                try:
                    calibration.stop_calibration()
                except Exception as e:
                    print(f"Error stopping calibration: {e}")
                    import traceback
                    traceback.print_exc()
        
        # Start/stop PLR test
        if key == ord('l'):
            if not plr.plr_test_active:
                if not et.overlay_running:
                    print("Please enable gaze overlay (press 'g') before starting PLR test.")
                else:
                    plr.start_plr_test()
            else:
                try:
                    plr.stop_plr_test()
                except Exception as e:
                    print(f"Error stopping PLR test: {e}")
                    import traceback
                    traceback.print_exc()
        
        # Start automated test suite
        if key == ord('z'):
            if automated_suite_running:
                print("Automated test suite is already running!")
            elif not et.overlay_running:
                print("Please enable gaze overlay (press 'g') before starting automated test suite.")
            else:
                automated_suite_running = True
                automated_suite_state = 'initial_calibration'
                automated_suite_results = {}
                automated_suite_test_start_time = None
                # Store initial blink count to calculate blinks during suite only
                automated_suite_initial_blinks = et.blinks_detected
                automated_suite_calibration_active = True
                import time
                automated_suite_calibration_start_time = time.time()
                # Enable calibration overlay
                automated_suite_calibration_overlay_active = True
                print("\n" + "="*60)
                print("STARTING AUTOMATED TEST SUITE")
                print("="*60)
                print("\nCalibration: Look at the center dot for 5 seconds...")
        
        # Handle automated test suite progression
        if automated_suite_running:
            import time
            import json
            from datetime import datetime
            
            # Initial calibration or between-test calibration
            if automated_suite_state in ['initial_calibration', 'calibration_before_saccade', 
                                         'calibration_before_pursuit', 'calibration_before_fixed_point', 
                                         'calibration_before_plr']:
                if automated_suite_calibration_start_time is None:
                    automated_suite_calibration_start_time = time.time()
                    automated_suite_calibration_active = True
                    automated_suite_calibration_overlay_active = True
                
                elapsed = time.time() - automated_suite_calibration_start_time
                
                if elapsed >= 5.0:  # 5 seconds elapsed
                    # Perform calibration
                    if et.raw_gaze_x is not None and et.raw_gaze_y is not None:
                        et.calibrate_gaze(et.raw_gaze_x, et.raw_gaze_y)
                        print("Calibration complete!")
                    else:
                        print("Warning: No gaze data available for calibration")
                    
                    # Move to next state
                    automated_suite_calibration_active = False
                    automated_suite_calibration_overlay_active = False
                    automated_suite_calibration_start_time = None
                    
                    if automated_suite_state == 'initial_calibration':
                        print("\n1/4: Starting Saccade Test...")
                        automated_suite_state = 'saccade'
                        saccade_test.start_saccade_test()
                    elif automated_suite_state == 'calibration_before_saccade':
                        print("\n1/4: Starting Saccade Test...")
                        automated_suite_state = 'saccade'
                        saccade_test.start_saccade_test()
                    elif automated_suite_state == 'calibration_before_pursuit':
                        print("\n2/4: Starting Smooth Pursuit Test...")
                        automated_suite_state = 'pursuit'
                        smooth_pursuit.start_smooth_pursuit_test()
                    elif automated_suite_state == 'calibration_before_fixed_point':
                        print("\n3/4: Starting Fixed Point Stability Test...")
                        automated_suite_state = 'fixed_point'
                        fixed_point_stability.start_fixed_point_test()
                    elif automated_suite_state == 'calibration_before_plr':
                        print("\n4/4: Starting PLR Test...")
                        automated_suite_state = 'plr'
                        plr.start_plr_test()
            
            # Saccade test
            elif automated_suite_state == 'saccade':
                if not saccade_test.saccade_testing_active:
                    # Saccade test completed - try to get results from JSON file
                    # Wait a moment for file to be written
                    import time
                    time.sleep(0.5)  # Give file system time to write
                    
                    saccade_results = get_latest_test_results('saccade')
                    if saccade_results:
                        automated_suite_results['saccade'] = {}
                        if saccade_results.get('normal_saccade'):
                            normal = saccade_results['normal_saccade']
                            automated_suite_results['saccade']['normal'] = {
                                'total_saccades': normal.get('total_saccades'),
                                'valid_saccades': normal.get('valid_saccades'),
                                'average_latency_ms': normal.get('average_latency_ms'),
                                'std_latency_ms': normal.get('std_latency_ms'),
                                'average_velocity_deg_per_ms': normal.get('average_velocity_deg_per_ms'),
                                'std_velocity_deg_per_ms': normal.get('std_velocity_deg_per_ms'),
                                'average_accuracy_percent': normal.get('average_accuracy_percent'),
                                'std_accuracy_percent': normal.get('std_accuracy_percent')
                            }
                        if saccade_results.get('antisaccade'):
                            anti = saccade_results['antisaccade']
                            automated_suite_results['saccade']['antisaccade'] = {
                                'total_trials': anti.get('total_trials'),
                                'error_count': anti.get('error_count'),
                                'correct_count': anti.get('correct_count'),
                                'error_rate_percent': anti.get('error_rate_percent')
                            }
                    else:
                        print("Warning: Could not retrieve saccade test results from JSON file")
                    # Start calibration before next test
                    automated_suite_state = 'calibration_before_pursuit'
                    automated_suite_calibration_start_time = time.time()
                    automated_suite_calibration_active = True
                    automated_suite_calibration_overlay_active = True
                    print("\nCalibration: Look at the center dot for 5 seconds...")
            
            # Smooth pursuit test
            elif automated_suite_state == 'pursuit':
                if not smooth_pursuit.smooth_pursuit_active:
                    # Pursuit test completed - try to get results from JSON file
                    import time
                    time.sleep(0.5)  # Give file system time to write
                    
                    pursuit_results = get_latest_test_results('smooth_pursuit')
                    if pursuit_results:
                        automated_suite_results['smooth_pursuit'] = {
                            'total_measurements': pursuit_results.get('total_measurements'),
                            'horizontal_measurements': pursuit_results.get('horizontal_measurements'),
                            'vertical_measurements': pursuit_results.get('vertical_measurements'),
                            'average_gain': pursuit_results.get('average_gain'),
                            'std_gain': pursuit_results.get('std_gain'),
                            'average_gain_horizontal': pursuit_results.get('average_gain_horizontal'),
                            'average_gain_vertical': pursuit_results.get('average_gain_vertical'),
                            'average_latency_ms': pursuit_results.get('average_latency_ms'),
                            'std_latency_ms': pursuit_results.get('std_latency_ms'),
                            'average_latency_horizontal_ms': pursuit_results.get('average_latency_horizontal_ms'),
                            'average_latency_vertical_ms': pursuit_results.get('average_latency_vertical_ms')
                        }
                    else:
                        print("Warning: Could not retrieve smooth pursuit test results from JSON file")
                    # Start calibration before next test
                    automated_suite_state = 'calibration_before_fixed_point'
                    automated_suite_calibration_start_time = time.time()
                    automated_suite_calibration_active = True
                    automated_suite_calibration_overlay_active = True
                    print("\nCalibration: Look at the center dot for 5 seconds...")
            
            # Fixed point stability test
            elif automated_suite_state == 'fixed_point':
                if not fixed_point_stability.fixed_point_active:
                    # Fixed point test completed - try to get results from JSON file
                    import time
                    time.sleep(0.5)  # Give file system time to write
                    
                    stability_results = get_latest_test_results('fixed_point')
                    if stability_results:
                        automated_suite_results['fixed_point'] = {
                            'total_measurements': stability_results.get('total_measurements'),
                            'average_deviation_degrees': stability_results.get('average_deviation_degrees'),
                            'median_deviation_degrees': stability_results.get('median_deviation_degrees'),
                            'rms_deviation_degrees': stability_results.get('rms_deviation_degrees'),
                            'std_deviation_degrees': stability_results.get('std_deviation_degrees'),
                            'max_deviation_degrees': stability_results.get('max_deviation_degrees'),
                            'min_deviation_degrees': stability_results.get('min_deviation_degrees')
                        }
                    else:
                        print("Warning: Could not retrieve fixed point stability test results from JSON file")
                    # Start calibration before next test
                    automated_suite_state = 'calibration_before_plr'
                    automated_suite_calibration_start_time = time.time()
                    automated_suite_calibration_active = True
                    automated_suite_calibration_overlay_active = True
                    print("\nCalibration: Look at the center dot for 5 seconds...")
            
            # PLR test
            elif automated_suite_state == 'plr':
                if not plr.plr_test_active:
                    # PLR test completed - try to get results from JSON file
                    import time
                    time.sleep(0.5)  # Give file system time to write
                    
                    plr_results = get_latest_test_results('plr')
                    if plr_results:
                        automated_suite_results['plr'] = {
                            'baseline_samples_count': plr_results.get('baseline_samples_count'),
                            'response_samples_count': plr_results.get('response_samples_count'),
                            'baseline_diameter_pixels': plr_results.get('baseline_diameter_pixels'),
                            'min_response_diameter_pixels': plr_results.get('min_response_diameter_pixels'),
                            'constriction_amplitude_pixels': plr_results.get('constriction_amplitude_pixels'),
                            'constriction_amplitude_percent': plr_results.get('constriction_amplitude_percent'),
                            'plr_latency_ms': plr_results.get('plr_latency_ms')
                        }
                    else:
                        print("Warning: Could not retrieve PLR test results from JSON file")
                    
                    # Get blink rate statistics
                    blink_rate_1min = et.get_blink_rate(60)
                    blink_rate_30sec = et.get_blink_rate(30)
                    
                    # Calculate blinks during the automated suite only
                    blinks_during_suite = et.blinks_detected - automated_suite_initial_blinks
                    
                    automated_suite_results['blink_rate'] = {
                        'blinks_per_minute_60s': blink_rate_1min,
                        'blinks_per_minute_30s': blink_rate_30sec,
                        'total_blinks': blinks_during_suite
                    }
                    
                    # Save combined results
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    combined_filename = f"combined_test_results_{timestamp}.json"
                    
                    all_results = {
                        'metadata': {
                            'timestamp': timestamp,
                            'test_date': datetime.now().isoformat(),
                            'monitor_width': et.monitor_width,
                            'monitor_height': et.monitor_height,
                            'tests_run': ['saccade', 'smooth_pursuit', 'fixed_point', 'plr'],
                            'test_order': 'saccade → smooth_pursuit → fixed_point → plr'
                        },
                        'results': automated_suite_results
                    }
                    
                    try:
                        with open(combined_filename, 'w') as f:
                            json.dump(all_results, f, indent=2)
                        
                        # Generate HTML report
                        html_filename = combined_filename.replace('.json', '.html')
                        report_generator.generate_html_report(all_results, html_filename)
                        
                        print("\n" + "="*60)
                        print("AUTOMATED TEST SUITE COMPLETE")
                        print("="*60)
                        print(f"Combined results saved to: {combined_filename}")
                        print(f"HTML report saved to: {html_filename}")
                        print("\nSummary of Results:")
                        if 'saccade' in automated_suite_results:
                            if 'normal' in automated_suite_results['saccade']:
                                normal = automated_suite_results['saccade']['normal']
                                print(f"  Saccade Latency: {normal.get('average_latency_ms', 0):.2f} ms (std: {normal.get('std_latency_ms', 0):.2f})")
                                print(f"  Saccade Velocity: {normal.get('average_velocity_deg_per_ms', 0):.4f} deg/ms (std: {normal.get('std_velocity_deg_per_ms', 0):.4f})")
                                print(f"  Saccade Accuracy: {normal.get('average_accuracy_percent', 0):.2f}% (std: {normal.get('std_accuracy_percent', 0):.2f})")
                            if 'antisaccade' in automated_suite_results['saccade']:
                                anti = automated_suite_results['saccade']['antisaccade']
                                print(f"  Antisaccade Error Rate: {anti.get('error_rate_percent', 0):.2f}% ({anti.get('error_count', 0)}/{anti.get('total_trials', 0)})")
                        if 'smooth_pursuit' in automated_suite_results:
                            pursuit = automated_suite_results['smooth_pursuit']
                            print(f"  Smooth Pursuit Gain: {pursuit.get('average_gain', 0):.3f} (std: {pursuit.get('std_gain', 0):.3f})")
                            print(f"  Smooth Pursuit Latency: {pursuit.get('average_latency_ms', 0):.2f} ms (std: {pursuit.get('std_latency_ms', 0):.2f})")
                        if 'fixed_point' in automated_suite_results:
                            fixed = automated_suite_results['fixed_point']
                            print(f"  Fixed Point Deviation: {fixed.get('average_deviation_degrees', 0):.4f}° (RMS: {fixed.get('rms_deviation_degrees', 0):.4f}°)")
                        if 'plr' in automated_suite_results:
                            plr_data = automated_suite_results['plr']
                            if plr_data.get('plr_latency_ms') is not None:
                                print(f"  PLR Latency: {plr_data.get('plr_latency_ms', 0):.2f} ms")
                            print(f"  PLR Constriction: {plr_data.get('constriction_amplitude_percent', 0):.2f}%")
                        if 'blink_rate' in automated_suite_results:
                            blink = automated_suite_results['blink_rate']
                            print(f"  Blink Rate: {blink.get('blinks_per_minute_60s', 0):.1f} blinks/min (Total: {blink.get('total_blinks', 0)})")
                        print("="*60 + "\n")
                    except Exception as e:
                        print(f"Error saving combined results: {e}")
                        import traceback
                        traceback.print_exc()
                    
                    # Reset automated suite state
                    automated_suite_running = False
                    automated_suite_state = None
                    automated_suite_results = {}
        
        # Display blink statistics
        if key == ord('b'):  # Press 'b' to show blink statistics
            et.print_blink_stats()
        
        if key == ord('q'):  # Press 'q' to quit
            et.stop_gaze_overlay()
            break   
        elif key == ord(' '):  # Press spacebar to start/stop
            while True:
                key = cv2.waitKey(1) & 0xFF
                if key == ord(' '):  # Press spacebar again to resume
                    break
                elif key == ord('q'):  # Press 'q' to quit
                    break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

