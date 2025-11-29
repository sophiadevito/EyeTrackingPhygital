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
    print("  B        - Show blink statistics")
    print("  Q        - Quit")
    print("")
    
    # We need to modify the eye tracking to handle test keys
    # Let's create a wrapper that handles this
    run_with_tests(args.debug, accuracy_test_update, saccade_test_update, pursuit_test_update, stability_test_update, calibration_update)

def run_with_tests(debug_calibration_flag, accuracy_update_callback, saccade_update_callback, pursuit_update_callback, stability_update_callback, calibration_update_callback):
    """
    Run eye tracking with all testing modules integration
    Handles keyboard input for all tests
    """
    import cv2
    import eye_tracking as et
    
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
        if calibration.plr_test_active:
            try:
                # Update PLR test with current pupil diameter
                calibration.update_plr_test(et.current_pupil_diameter)
                
                # Keep flash window responsive
                if calibration.plr_flash_window is not None:
                    try:
                        calibration.plr_flash_window.update_idletasks()
                        calibration.plr_flash_window.update()
                    except:
                        pass
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

