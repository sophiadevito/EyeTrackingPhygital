"""
Report Generator for Eye Tracking Test Results

Generates HTML reports from combined test results with color-coded metrics
based on configurable thresholds.
"""

import json
from datetime import datetime

# Define thresholds for each metric (user-configurable)
# Format: {'good': value, 'warning': value}
# For "lower is better" metrics (latency, deviation, error rate), values below 'good' are green
# For "higher is better" metrics (gain, accuracy), values above 'good' are green
THRESHOLDS = {
    'saccade_latency_ms': {'good': 350, 'warning': 500},  # Lower is better
    'saccade_velocity_deg_per_ms': {'good': 0.2, 'warning': 0.15},  # Higher is better
    'saccade_accuracy_percent': {'good': 50, 'warning': 30},  # Higher is better
    'antisaccade_error_rate_percent': {'good': 20, 'warning': 40},  # Lower is better
    'smooth_pursuit_gain': {'good': 0.4, 'warning': 0.3},  # Higher is better
    'smooth_pursuit_latency_ms': {'good': 350, 'warning': 450},  # Lower is better
    'fixed_point_deviation_degrees': {'good': 2.0, 'warning': 3.0},  # Lower is better
    'plr_latency_ms': {'good': 500, 'warning': 600},  # Lower is better
    'plr_constriction_percent': {'good': 15, 'warning': 10},  # Higher is better
    'blink_rate_per_min': {'good_min': 0, 'good_max': 10, 'warning_max': 20},  # Range: 0-15 good, 15-30 warning, 30+ poor
}

def get_color_for_value(value, thresholds, lower_is_better=False):
    """
    Return color class based on value and thresholds.
    
    Args:
        value: The metric value to evaluate
        thresholds: Dictionary with 'good' and 'warning' keys, or range keys for blink rate
        lower_is_better: If True, lower values are better (e.g., latency, error rate)
    
    Returns:
        'good', 'warning', 'poor', or 'gray' (if value is None)
    """
    if value is None:
        return 'gray'
    
    # Special handling for blink rate (range-based)
    if 'good_min' in thresholds and 'good_max' in thresholds:
        good_min = thresholds['good_min']
        good_max = thresholds['good_max']
        warning_max = thresholds.get('warning_max', good_max * 2)
        
        if good_min <= value <= good_max:
            return 'good'
        elif value <= warning_max:
            return 'warning'
        else:
            return 'poor'
    
    if lower_is_better:
        if value <= thresholds['good']:
            return 'good'
        elif value <= thresholds['warning']:
            return 'warning'
        else:
            return 'poor'
    else:
        if value >= thresholds['good']:
            return 'good'
        elif value >= thresholds['warning']:
            return 'warning'
        else:
            return 'poor'

def calculate_normalized_score(value, thresholds, lower_is_better=False):
    """
    Convert a metric value to a normalized score (0-100).
    
    Args:
        value: The metric value
        thresholds: Dictionary with 'good' and 'warning' keys, or range keys for blink rate
        lower_is_better: Whether lower values are better
    
    Returns:
        Score from 0-100, or None if value is None
    """
    if value is None:
        return None
    
    # Special handling for blink rate (range-based)
    if 'good_min' in thresholds and 'good_max' in thresholds:
        good_min = thresholds['good_min']
        good_max = thresholds['good_max']
        warning_max = thresholds.get('warning_max', good_max * 2)
        
        if good_min <= value <= good_max:
            # Good range: map to 100-67
            # Linear interpolation within the good range
            if good_max == good_min:
                return 100
            return 100 - ((value - good_min) / (good_max - good_min)) * 33
        elif value <= warning_max:
            # Warning range: map to 67-33
            if warning_max == good_max:
                return 50
            return 67 - ((value - good_max) / (warning_max - good_max)) * 34
        else:
            # Poor range: map to 33-0
            poor_max = max(warning_max * 1.5, value * 1.2)
            if poor_max == warning_max:
                return 0
            return max(0, 33 - ((value - warning_max) / (poor_max - warning_max)) * 33)
    
    good_val = thresholds['good']
    warning_val = thresholds['warning']
    
    if lower_is_better:
        if value <= good_val:
            # Good range: map to 100-67
            if good_val == 0:
                return 100 if value == 0 else 0
            return 100 - ((value / good_val) * 33)
        elif value <= warning_val:
            # Warning range: map to 67-33
            if warning_val == good_val:
                return 50
            return 67 - (((value - good_val) / (warning_val - good_val)) * 34)
        else:
            # Poor range: map to 33-0
            poor_max = max(warning_val * 2, value * 1.2)
            if poor_max == warning_val:
                return 0
            return max(0, 33 - (((value - warning_val) / (poor_max - warning_val)) * 33))
    else:
        if value >= good_val:
            # Good range: map to 100-67
            good_max = max(good_val * 1.5, value * 1.2)
            if good_max == good_val:
                return 100
            return 100 - (((value - good_val) / (good_max - good_val)) * 33)
        elif value >= warning_val:
            # Warning range: map to 67-33
            if good_val == warning_val:
                return 50
            return 67 - (((good_val - value) / (good_val - warning_val)) * 34)
        else:
            # Poor range: map to 33-0
            poor_min = min(0, warning_val * 0.5)
            if warning_val == poor_min:
                return 0
            return max(0, 33 - (((warning_val - value) / (warning_val - poor_min)) * 33))

def calculate_overall_score(results):
    """
    Calculate overall combined score from all metrics.
    
    Args:
        results: Dictionary containing test results
    
    Returns:
        Tuple of (overall_score, total_weight) where score is 0-100
    """
    scores = []
    weights = []
    
    # Define weights for each metric (higher = more important)
    metric_weights = {
        'saccade_latency_ms': 1.0,
        'saccade_velocity_deg_per_ms': 1.0,
        'saccade_accuracy_percent': 1.5,  # More important
        'antisaccade_error_rate_percent': 1.0,
        'smooth_pursuit_gain': 1.5,  # More important
        'smooth_pursuit_latency_ms': 1.0,
        'fixed_point_deviation_degrees': 1.0,
        'plr_latency_ms': 1.0,
        'plr_constriction_percent': 1.0,
        'blink_rate_per_min': 0.5,  # Less important
    }
    
    # Saccade metrics
    if 'saccade' in results:
        normal = results['saccade'].get('normal', {})
        if normal:
            # Latency
            latency = normal.get('average_latency_ms')
            if latency is not None:
                score = calculate_normalized_score(latency, THRESHOLDS['saccade_latency_ms'], lower_is_better=True)
                if score is not None:
                    scores.append(score)
                    weights.append(metric_weights['saccade_latency_ms'])
            
            # Velocity
            velocity = normal.get('average_velocity_deg_per_ms')
            if velocity is not None:
                score = calculate_normalized_score(velocity, THRESHOLDS['saccade_velocity_deg_per_ms'])
                if score is not None:
                    scores.append(score)
                    weights.append(metric_weights['saccade_velocity_deg_per_ms'])
            
            # Accuracy
            accuracy = normal.get('average_accuracy_percent')
            if accuracy is not None:
                score = calculate_normalized_score(accuracy, THRESHOLDS['saccade_accuracy_percent'])
                if score is not None:
                    scores.append(score)
                    weights.append(metric_weights['saccade_accuracy_percent'])
        
        anti = results['saccade'].get('antisaccade', {})
        if anti:
            error_rate = anti.get('error_rate_percent')
            if error_rate is not None:
                score = calculate_normalized_score(error_rate, THRESHOLDS['antisaccade_error_rate_percent'], lower_is_better=True)
                if score is not None:
                    scores.append(score)
                    weights.append(metric_weights['antisaccade_error_rate_percent'])
    
    # Smooth pursuit
    if 'smooth_pursuit' in results:
        pursuit = results['smooth_pursuit']
        gain = pursuit.get('average_gain')
        if gain is not None:
            score = calculate_normalized_score(gain, THRESHOLDS['smooth_pursuit_gain'])
            if score is not None:
                scores.append(score)
                weights.append(metric_weights['smooth_pursuit_gain'])
        
        latency = pursuit.get('average_latency_ms')
        if latency is not None and latency > 0:
            score = calculate_normalized_score(latency, THRESHOLDS['smooth_pursuit_latency_ms'], lower_is_better=True)
            if score is not None:
                scores.append(score)
                weights.append(metric_weights['smooth_pursuit_latency_ms'])
    
    # Fixed point
    if 'fixed_point' in results:
        fixed = results['fixed_point']
        deviation = fixed.get('average_deviation_degrees')
        if deviation is not None:
            score = calculate_normalized_score(deviation, THRESHOLDS['fixed_point_deviation_degrees'], lower_is_better=True)
            if score is not None:
                scores.append(score)
                weights.append(metric_weights['fixed_point_deviation_degrees'])
    
    # PLR
    if 'plr' in results:
        plr_data = results['plr']
        latency = plr_data.get('plr_latency_ms')
        if latency is not None:
            score = calculate_normalized_score(latency, THRESHOLDS['plr_latency_ms'], lower_is_better=True)
            if score is not None:
                scores.append(score)
                weights.append(metric_weights['plr_latency_ms'])
        
        constriction = plr_data.get('constriction_amplitude_percent')
        if constriction is not None:
            score = calculate_normalized_score(constriction, THRESHOLDS['plr_constriction_percent'])
            if score is not None:
                scores.append(score)
                weights.append(metric_weights['plr_constriction_percent'])
    
    # Blink rate
    if 'blink_rate' in results:
        blink = results['blink_rate']
        rate = blink.get('blinks_per_minute_60s')
        if rate is not None:
            score = calculate_normalized_score(rate, THRESHOLDS['blink_rate_per_min'])
            if score is not None:
                scores.append(score)
                weights.append(metric_weights['blink_rate_per_min'])
    
    # Calculate weighted average
    if not scores:
        return None, 0
    
    total_weight = sum(weights)
    if total_weight == 0:
        return None, 0
    
    weighted_sum = sum(score * weight for score, weight in zip(scores, weights))
    overall_score = weighted_sum / total_weight
    
    # Invert the score: High normalized scores (good performance) → Low dial scores (good)
    # Low normalized scores (poor performance) → High dial scores (poor)
    # This makes the dial show: Low numbers = Good, High numbers = Poor
    inverted_score = 100 - overall_score
    
    return inverted_score, total_weight

def create_score_dial_html(score):
    """
    Create HTML for a credit-score-style dial with concussion detection labels.
    
    Args:
        score: Overall score from 0-100
    
    Returns:
        HTML string for the dial
    """
    if score is None:
        score = 0
    
    # Convert score (0-100) to degrees (0-180 degrees for 180-degree arc)
    # Dial: 0° (right, green/good) to 180° (left, red/poor)
    # Inverted: High scores = Poor (left), Low scores = Good (right)
    # Score 0 (good) → 0° (right, green), Score 100 (poor) → 180° (left, red)
    angle = (score / 100) * 180
    needle_rotation = angle - 90
    
    # Determine color and message based on score (inverted logic)
    # High scores = Poor, Low scores = Good
    if score <= 33:  # Good range (low score)
        color = '#28a745'
        message = 'No concussion detected'
        label_class = 'good'
    elif score <= 67:  # Warning range (medium score)
        color = '#CC5500'
        message = 'Potential concussion detected'
        label_class = 'warning'
    else:  # Poor range (high score)
        color = '#dc3545'
        message = 'Concussion detected'
        label_class = 'poor'
    
    html = f"""
            <div class="score-dial-container">
                <div class="score-value">{score:.0f}</div>
                <div class="score-dial-wrapper">
                    <svg class="score-dial" viewBox="0 0 200 120" xmlns="http://www.w3.org/2000/svg">
                        <!-- Single 180-degree arc divided into three zones:
                             0-60° green, 60-120° yellow, 120-180° red
                             Arc center: (100, 100), radius: 80
                             Points: 0°(180,100-right), 60°(140,30.72), 120°(60,30.72), 180°(20,100-left) -->
                        <!-- Green zone: 0-60° (right third) -->
                        <path d="M 180 100 A 80 80 0 0 0 140 30.72" 
                              fill="none" 
                              stroke="#dc3545" 
                              stroke-width="12" 
                              stroke-linecap="round"
                              opacity="0.3"/>
                        <!-- Yellow zone: 60-120° (middle third) -->
                        <path d="M 140 30.72 A 80 80 0 0 0 60 30.72" 
                              fill="none" 
                              stroke="#ffc107" 
                              stroke-width="12" 
                              stroke-linecap="round"
                              opacity="0.3"/>
                        <!-- Red zone: 120-180° (left third) -->
                        <path d="M 60 30.72 A 80 80 0 0 0 20 100" 
                              fill="none" 
                              stroke="#28a745"
                              stroke-width="12" 
                              stroke-linecap="round"
                              opacity="0.3"/>
                        
                        <!-- Needle -->
                        <g transform="translate(100, 100) rotate({needle_rotation})">
                            <line x1="0" y1="0" x2="0" y2="-75" 
                                  stroke="{color}" 
                                  stroke-width="3" 
                                  stroke-linecap="round"/>
                            <circle cx="0" cy="0" r="4" fill="{color}"/>
                        </g>
                    </svg>
                </div>
                <div class="score-message {label_class}">{message}</div>
            </div>
    """
    return html

def smooth_gaze_path(gaze_path, window_size=5):
    """
    Apply moving average smoothing to gaze path to reduce jitter.
    
    Args:
        gaze_path: List of [timestamp, x, y] tuples
        window_size: Number of points to average (default 5)
    
    Returns:
        Smoothed gaze path in same format
    """
    if len(gaze_path) < window_size:
        return gaze_path
    
    smoothed = []
    for i in range(len(gaze_path)):
        # Define window boundaries
        start = max(0, i - window_size // 2)
        end = min(len(gaze_path), i + window_size // 2 + 1)
        window = gaze_path[start:end]
        
        # Calculate average position
        avg_x = sum(p[1] for p in window) / len(window)
        avg_y = sum(p[2] for p in window) / len(window)
        
        # Keep original timestamp
        smoothed.append([gaze_path[i][0], avg_x, avg_y])
    
    return smoothed

def create_eye_path_visualization(gaze_path, target_data, monitor_width, monitor_height):
    """
    Create an SVG visualization of the eye path during saccade testing.
    
    Args:
        gaze_path: List of [timestamp, x, y] tuples
        target_data: List of target data dictionaries with target_x, target_y
        monitor_width: Screen width in pixels
        monitor_height: Screen height in pixels
    
    Returns:
        HTML string with SVG visualization
    """
    if not gaze_path or len(gaze_path) < 2:
        return '<p style="color: #6c757d; font-style: italic;">No gaze path data available for visualization.</p>'
    
    # Filter out points outside screen bounds (with small margin to catch edge errors)
    margin = 10  # pixels margin from screen edges
    filtered_gaze_path = [
        point for point in gaze_path
        if len(point) >= 3 and 
           margin <= point[1] <= (monitor_width - margin) and
           margin <= point[2] <= (monitor_height - margin)
    ]
    
    # Use filtered path, but fall back to original if filtering removed too many points
    if len(filtered_gaze_path) < 2:
        filtered_gaze_path = gaze_path  # Use original if filtering was too aggressive
    
    # Apply smoothing to reduce jitter
    filtered_gaze_path = smooth_gaze_path(filtered_gaze_path, window_size=5)
    
    # Extract coordinates (use filtered and smoothed path)
    gaze_x = [point[1] for point in filtered_gaze_path]
    gaze_y = [point[2] for point in filtered_gaze_path]
    
    # Calculate bounds with padding
    min_x, max_x = min(gaze_x), max(gaze_x)
    min_y, max_y = min(gaze_y), max(gaze_y)
    
    # Add padding (10% on each side)
    x_range = max_x - min_x
    y_range = max_y - min_y
    padding_x = max(x_range * 0.1, 50)
    padding_y = max(y_range * 0.1, 50)
    
    plot_min_x = max(0, min_x - padding_x)
    plot_max_x = min(monitor_width, max_x + padding_x)
    plot_min_y = max(0, min_y - padding_y)
    plot_max_y = min(monitor_height, max_y + padding_y)
    
    plot_width = plot_max_x - plot_min_x
    plot_height = plot_max_y - plot_min_y
    
    # SVG dimensions (scaled to fit nicely in report)
    svg_width = 800
    svg_height = 600
    
    # Scale factors
    scale_x = svg_width / plot_width if plot_width > 0 else 1
    scale_y = svg_height / plot_height if plot_height > 0 else 1
    
    # Normalize coordinates to SVG space
    def normalize_x(x):
        return (x - plot_min_x) * scale_x
    
    def normalize_y(y):
        return (y - plot_min_y) * scale_y
    
    # Create path string for gaze trajectory
    path_points = []
    for i, point in enumerate(filtered_gaze_path):
        x = normalize_x(point[1])
        y = normalize_y(point[2])
        if i == 0:
            path_points.append(f"M {x:.2f} {y:.2f}")
        else:
            path_points.append(f"L {x:.2f} {y:.2f}")
    
    path_d = " ".join(path_points)
    
    # Get target positions
    target_positions = []
    if target_data:
        for target in target_data:
            if 'target_x' in target and 'target_y' in target:
                target_positions.append((target['target_x'], target['target_y']))
    
    # Create SVG
    html = f"""
            <div class="eye-path-visualization">
                <svg width="{svg_width}" height="{svg_height}" viewBox="0 0 {svg_width} {svg_height}" 
                     style="border: 1px solid #dee2e6; border-radius: 8px; background: #f8f9fa;">
                    <!-- Background grid (optional) -->
                    <defs>
                        <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
                            <path d="M 40 0 L 0 0 0 40" fill="none" stroke="#e9ecef" stroke-width="1"/>
                        </pattern>
                    </defs>
                    <rect width="100%" height="100%" fill="url(#grid)" opacity="0.3"/>
                    
                    <!-- Gaze path -->
                    <path d="{path_d}" 
                          fill="none" 
                          stroke="#667eea" 
                          stroke-width="2" 
                          stroke-opacity="0.7"
                          stroke-linecap="round"
                          stroke-linejoin="round"/>
                    
                    <!-- Gaze points (small dots) -->
                    {''.join([f'<circle cx="{normalize_x(x):.2f}" cy="{normalize_y(y):.2f}" r="2" fill="#667eea" opacity="0.5"/>' 
                              for x, y in zip(gaze_x, gaze_y)])}
                    
                    <!-- Target positions -->
                    {''.join([f'<circle cx="{normalize_x(tx):.2f}" cy="{normalize_y(ty):.2f}" r="8" fill="#dc3545" stroke="#fff" stroke-width="2"/>' 
                              for tx, ty in target_positions])}
                    
                    <!-- Start point marker -->
                    {f'<circle cx="{normalize_x(gaze_x[0]):.2f}" cy="{normalize_y(gaze_y[0]):.2f}" r="6" fill="#28a745" stroke="#fff" stroke-width="2"/>' if gaze_x else ''}
                    
                    <!-- End point marker -->
                    {f'<circle cx="{normalize_x(gaze_x[-1]):.2f}" cy="{normalize_y(gaze_y[-1]):.2f}" r="6" fill="#ffc107" stroke="#fff" stroke-width="2"/>' if gaze_x else ''}
                </svg>
                <div class="eye-path-legend">
                    <div class="legend-item">
                        <span class="legend-color" style="background: #667eea;"></span>
                        <span>Eye Path</span>
                    </div>
                    <div class="legend-item">
                        <span class="legend-color" style="background: #28a745;"></span>
                        <span>Start</span>
                    </div>
                    <div class="legend-item">
                        <span class="legend-color" style="background: #ffc107;"></span>
                        <span>End</span>
                    </div>
                    <div class="legend-item">
                        <span class="legend-color" style="background: #dc3545;"></span>
                        <span>Targets</span>
                    </div>
                </div>
            </div>
    """
    return html

def create_pursuit_path_visualization(gaze_path, target_path, monitor_width, monitor_height):
    """
    Create an SVG visualization of the eye path during smooth pursuit testing.
    Shows both the target path and the gaze path.
    
    Args:
        gaze_path: List of [timestamp, x, y] tuples
        target_path: List of target positions (if available)
        monitor_width: Screen width in pixels
        monitor_height: Screen height in pixels
    
    Returns:
        HTML string with SVG visualization
    """
    if not gaze_path or len(gaze_path) < 2:
        return '<p style="color: #6c757d; font-style: italic;">No gaze path data available for visualization.</p>'
    
    # Filter out points outside screen bounds (with small margin to catch edge errors)
    margin = 10  # pixels margin from screen edges
    filtered_gaze_path = [
        point for point in gaze_path
        if len(point) >= 3 and 
           margin <= point[1] <= (monitor_width - margin) and
           margin <= point[2] <= (monitor_height - margin)
    ]
    
    # Use filtered path, but fall back to original if filtering removed too many points
    if len(filtered_gaze_path) < 2:
        filtered_gaze_path = gaze_path  # Use original if filtering was too aggressive
    
    # Apply smoothing to reduce jitter
    filtered_gaze_path = smooth_gaze_path(filtered_gaze_path, window_size=5)
    
    # Extract coordinates (use filtered and smoothed path)
    gaze_x = [point[1] for point in filtered_gaze_path]
    gaze_y = [point[2] for point in filtered_gaze_path]
    
    # Calculate bounds with padding
    min_x, max_x = min(gaze_x), max(gaze_x)
    min_y, max_y = min(gaze_y), max(gaze_y)
    
    # Add padding (10% on each side)
    x_range = max_x - min_x
    y_range = max_y - min_y
    padding_x = max(x_range * 0.1, 50)
    padding_y = max(y_range * 0.1, 50)
    
    plot_min_x = max(0, min_x - padding_x)
    plot_max_x = min(monitor_width, max_x + padding_x)
    plot_min_y = max(0, min_y - padding_y)
    plot_max_y = min(monitor_height, max_y + padding_y)
    
    plot_width = plot_max_x - plot_min_x
    plot_height = plot_max_y - plot_min_y
    
    # SVG dimensions
    svg_width = 800
    svg_height = 600
    
    # Scale factors
    scale_x = svg_width / plot_width if plot_width > 0 else 1
    scale_y = svg_height / plot_height if plot_height > 0 else 1
    
    # Normalize coordinates to SVG space
    def normalize_x(x):
        return (x - plot_min_x) * scale_x
    
    def normalize_y(y):
        return (y - plot_min_y) * scale_y
    
    # Create path string for gaze trajectory
    path_points = []
    for i, point in enumerate(filtered_gaze_path):
        x = normalize_x(point[1])
        y = normalize_y(point[2])
        if i == 0:
            path_points.append(f"M {x:.2f} {y:.2f}")
        else:
            path_points.append(f"L {x:.2f} {y:.2f}")
    
    path_d = " ".join(path_points)
    
    # Create SVG
    html = f"""
            <div class="eye-path-visualization">
                <svg width="{svg_width}" height="{svg_height}" viewBox="0 0 {svg_width} {svg_height}" 
                     style="border: 1px solid #dee2e6; border-radius: 8px; background: #f8f9fa;">
                    <!-- Background grid -->
                    <defs>
                        <pattern id="grid-pursuit" width="40" height="40" patternUnits="userSpaceOnUse">
                            <path d="M 40 0 L 0 0 0 40" fill="none" stroke="#e9ecef" stroke-width="1"/>
                        </pattern>
                    </defs>
                    <rect width="100%" height="100%" fill="url(#grid-pursuit)" opacity="0.3"/>
                    
                    <!-- Gaze path -->
                    <path d="{path_d}" 
                          fill="none" 
                          stroke="#667eea" 
                          stroke-width="2" 
                          stroke-opacity="0.7"
                          stroke-linecap="round"
                          stroke-linejoin="round"/>
                    
                    <!-- Gaze points -->
                    {''.join([f'<circle cx="{normalize_x(x):.2f}" cy="{normalize_y(y):.2f}" r="2" fill="#667eea" opacity="0.5"/>' 
                              for x, y in zip(gaze_x, gaze_y)])}
                    
                    <!-- Start point marker -->
                    {f'<circle cx="{normalize_x(gaze_x[0]):.2f}" cy="{normalize_y(gaze_y[0]):.2f}" r="6" fill="#28a745" stroke="#fff" stroke-width="2"/>' if gaze_x else ''}
                    
                    <!-- End point marker -->
                    {f'<circle cx="{normalize_x(gaze_x[-1]):.2f}" cy="{normalize_y(gaze_y[-1]):.2f}" r="6" fill="#ffc107" stroke="#fff" stroke-width="2"/>' if gaze_x else ''}
                </svg>
                <div class="eye-path-legend">
                    <div class="legend-item">
                        <span class="legend-color" style="background: #667eea;"></span>
                        <span>Gaze Path</span>
                    </div>
                    <div class="legend-item">
                        <span class="legend-color" style="background: #28a745;"></span>
                        <span>Start</span>
                    </div>
                    <div class="legend-item">
                        <span class="legend-color" style="background: #ffc107;"></span>
                        <span>End</span>
                    </div>
                </div>
            </div>
    """
    return html

def create_fixed_point_path_visualization(gaze_path, center_x, center_y, monitor_width, monitor_height):
    """
    Create an SVG visualization of the eye path during fixed point stability testing.
    Shows the gaze path around the center target.
    
    Args:
        gaze_path: List of [timestamp, x, y] tuples
        center_x: Center target X position
        center_y: Center target Y position
        monitor_width: Screen width in pixels
        monitor_height: Screen height in pixels
    
    Returns:
        HTML string with SVG visualization
    """
    if not gaze_path or len(gaze_path) < 2:
        return '<p style="color: #6c757d; font-style: italic;">No gaze path data available for visualization.</p>'
    
    # Filter out points outside screen bounds (with small margin to catch edge errors)
    margin = 10  # pixels margin from screen edges
    filtered_gaze_path = [
        point for point in gaze_path
        if len(point) >= 3 and 
           margin <= point[1] <= (monitor_width - margin) and
           margin <= point[2] <= (monitor_height - margin)
    ]
    
    # Use filtered path, but fall back to original if filtering removed too many points
    if len(filtered_gaze_path) < 2:
        filtered_gaze_path = gaze_path  # Use original if filtering was too aggressive
    
    # Apply smoothing to reduce jitter
    filtered_gaze_path = smooth_gaze_path(filtered_gaze_path, window_size=5)
    
    # Extract coordinates (use filtered and smoothed path)
    gaze_x = [point[1] for point in filtered_gaze_path]
    gaze_y = [point[2] for point in filtered_gaze_path]
    
    # Calculate bounds centered around the target
    all_x = gaze_x + [center_x]
    all_y = gaze_y + [center_y]
    
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    
    # Add padding (20% on each side to show deviation clearly)
    x_range = max_x - min_x
    y_range = max_y - min_y
    padding_x = max(x_range * 0.2, 100)
    padding_y = max(y_range * 0.2, 100)
    
    plot_min_x = max(0, min_x - padding_x)
    plot_max_x = min(monitor_width, max_x + padding_x)
    plot_min_y = max(0, min_y - padding_y)
    plot_max_y = min(monitor_height, max_y + padding_y)
    
    plot_width = plot_max_x - plot_min_x
    plot_height = plot_max_y - plot_min_y
    
    # SVG dimensions
    svg_width = 800
    svg_height = 600
    
    # Scale factors
    scale_x = svg_width / plot_width if plot_width > 0 else 1
    scale_y = svg_height / plot_height if plot_height > 0 else 1
    
    # Normalize coordinates to SVG space
    def normalize_x(x):
        return (x - plot_min_x) * scale_x
    
    def normalize_y(y):
        return (y - plot_min_y) * scale_y
    
    # Create path string for gaze trajectory
    path_points = []
    for i, point in enumerate(filtered_gaze_path):
        x = normalize_x(point[1])
        y = normalize_y(point[2])
        if i == 0:
            path_points.append(f"M {x:.2f} {y:.2f}")
        else:
            path_points.append(f"L {x:.2f} {y:.2f}")
    
    path_d = " ".join(path_points)
    
    # Create SVG
    html = f"""
            <div class="eye-path-visualization">
                <svg width="{svg_width}" height="{svg_height}" viewBox="0 0 {svg_width} {svg_height}" 
                     style="border: 1px solid #dee2e6; border-radius: 8px; background: #f8f9fa;">
                    <!-- Background grid -->
                    <defs>
                        <pattern id="grid-fixed" width="40" height="40" patternUnits="userSpaceOnUse">
                            <path d="M 40 0 L 0 0 0 40" fill="none" stroke="#e9ecef" stroke-width="1"/>
                        </pattern>
                    </defs>
                    <rect width="100%" height="100%" fill="url(#grid-fixed)" opacity="0.3"/>
                    
                    <!-- Center target -->
                    <circle cx="{normalize_x(center_x):.2f}" cy="{normalize_y(center_y):.2f}" r="10" fill="#dc3545" stroke="#fff" stroke-width="2"/>
                    <circle cx="{normalize_x(center_x):.2f}" cy="{normalize_y(center_y):.2f}" r="3" fill="#fff"/>
                    
                    <!-- Gaze path -->
                    <path d="{path_d}" 
                          fill="none" 
                          stroke="#667eea" 
                          stroke-width="2" 
                          stroke-opacity="0.7"
                          stroke-linecap="round"
                          stroke-linejoin="round"/>
                    
                    <!-- Gaze points -->
                    {''.join([f'<circle cx="{normalize_x(x):.2f}" cy="{normalize_y(y):.2f}" r="2" fill="#667eea" opacity="0.5"/>' 
                              for x, y in zip(gaze_x, gaze_y)])}
                    
                    <!-- Start point marker -->
                    {f'<circle cx="{normalize_x(gaze_x[0]):.2f}" cy="{normalize_y(gaze_y[0]):.2f}" r="6" fill="#28a745" stroke="#fff" stroke-width="2"/>' if gaze_x else ''}
                    
                    <!-- End point marker -->
                    {f'<circle cx="{normalize_x(gaze_x[-1]):.2f}" cy="{normalize_y(gaze_y[-1]):.2f}" r="6" fill="#ffc107" stroke="#fff" stroke-width="2"/>' if gaze_x else ''}
                </svg>
                <div class="eye-path-legend">
                    <div class="legend-item">
                        <span class="legend-color" style="background: #667eea;"></span>
                        <span>Gaze Path</span>
                    </div>
                    <div class="legend-item">
                        <span class="legend-color" style="background: #dc3545;"></span>
                        <span>Target Center</span>
                    </div>
                    <div class="legend-item">
                        <span class="legend-color" style="background: #28a745;"></span>
                        <span>Start</span>
                    </div>
                    <div class="legend-item">
                        <span class="legend-color" style="background: #ffc107;"></span>
                        <span>End</span>
                    </div>
                </div>
            </div>
    """
    return html

def create_metric_html(name, value, unit, thresholds, color, lower_is_better=False, details=None):
    """
    Create HTML for a single metric card.
    
    Args:
        name: Metric name
        value: Metric value
        unit: Unit of measurement
        thresholds: Dictionary with 'good' and 'warning' keys
        color: Color class ('good', 'warning', 'poor', 'gray')
        lower_is_better: Whether lower values are better
        details: Optional additional details text
    
    Returns:
        HTML string for the metric card
    """
    # Calculate scale percentage for visual bar
    # All scales should show: Good (left) → Warning (middle) → Poor (right)
    # The scale background shows: green (0-33%), yellow (33-66%), red (66-100%)
    # The bar position indicates where the value falls on this scale
    
    # Special handling for range-based thresholds (blink rate)
    if 'good_min' in thresholds and 'good_max' in thresholds:
        good_min = thresholds['good_min']
        good_max = thresholds['good_max']
        warning_max = thresholds.get('warning_max', good_max * 2)
        poor_max = max(warning_max * 1.5, value * 1.2)
        
        if good_min <= value <= good_max:
            # Good: map to 0-33%
            scale_percent = ((value - good_min) / (good_max - good_min)) * 33 if good_max > good_min else 0
        elif value <= warning_max:
            # Warning: map to 33-66%
            scale_percent = 33 + ((value - good_max) / (warning_max - good_max)) * 33 if warning_max > good_max else 50
        else:
            # Poor: map to 66-100%
            scale_percent = 66 + min(34, ((value - warning_max) / (poor_max - warning_max)) * 34) if poor_max > warning_max else 100
    else:
        good_val = thresholds['good']
        warning_val = thresholds['warning']
        
        if lower_is_better:
            # For lower is better: good (low) = left, poor (high) = right
            # Define a reasonable maximum for poor values to establish scale range
            poor_max = max(warning_val * 2, value * 1.5, good_val * 3)
            
            if value <= good_val:
                # Good: map to 0-33% (left side)
                # Best case: value = 0 or very low → 0%
                # Worst good case: value = good_val → 33%
                scale_percent = (value / good_val) * 33 if good_val > 0 else 0
            elif value <= warning_val:
                # Warning: map to 33-66% (middle)
                # value = good_val → 33%, value = warning_val → 66%
                scale_percent = 33 + ((value - good_val) / (warning_val - good_val)) * 33 if warning_val > good_val else 50
            else:
                # Poor: map to 66-100% (right side)
                # value = warning_val → 66%, value = poor_max → 100%
                scale_percent = 66 + min(34, ((value - warning_val) / (poor_max - warning_val)) * 34) if poor_max > warning_val else 100
        else:
            # For higher is better: good (high) = left, poor (low) = right
            # Define a reasonable minimum for poor values to establish scale range
            poor_min = min(0, warning_val * 0.5) if warning_val > 0 else 0
            # Define a reasonable maximum for good values
            good_max = max(good_val * 1.5, value * 1.2) if value > good_val else good_val * 1.5
            
            if value >= good_val:
                # Good: map to 0-33% (left side)
                # Best case: value = good_max or very high → 0%
                # Worst good case: value = good_val → 33%
                scale_percent = ((good_max - value) / (good_max - good_val)) * 33 if good_max > good_val else 0
            elif value >= warning_val:
                # Warning: map to 33-66% (middle)
                # value = good_val → 33%, value = warning_val → 66%
                scale_percent = 33 + ((good_val - value) / (good_val - warning_val)) * 33 if good_val > warning_val else 50
            else:
                # Poor: map to 66-100% (right side)
                # value = warning_val → 66%, value = poor_min → 100%
                scale_percent = 66 + min(34, ((warning_val - value) / (warning_val - poor_min)) * 34) if warning_val > poor_min else 100
    
    # Clamp to 0-100%
    scale_percent = max(0, min(100, scale_percent))
    
    status_label = color.capitalize()
    
    # Format value display
    if isinstance(value, float):
        value_str = f"{value:.2f}"
    elif isinstance(value, int):
        value_str = f"{value}"
    else:
        value_str = str(value)
    
    html = f"""
            <div class="metric-card {color}">
                <div class="metric-header">
                    <div class="metric-name">{name}</div>
                    <div class="metric-badge {color}">{status_label}</div>
                </div>
                <div class="metric-value">
                    {value_str}<span class="metric-unit">{unit}</span>
                </div>
                <div class="metric-scale">
                    <div class="metric-scale-bar" style="width: {scale_percent}%"></div>
                </div>
"""
    if details:
        html += f'                <div class="metric-details">{details}</div>\n'
    
    html += '            </div>\n'
    return html

def format_datetime(iso_string):
    """Format ISO datetime string for display"""
    try:
        if 'T' in iso_string:
            date_part, time_part = iso_string.split('T')
            time_part = time_part.split('.')[0]  # Remove microseconds
            return date_part, time_part
        return iso_string, ''
    except:
        return iso_string, ''

def generate_html_report(combined_results, output_filename=None):
    """
    Generate an HTML report from combined test results.
    
    Args:
        combined_results: Dictionary containing test results and metadata
        output_filename: Optional output filename (default: test_report_[timestamp].html)
    
    Returns:
        Filename of generated report
    """
    import glob
    import os
    
    if output_filename is None:
        timestamp = combined_results['metadata'].get('timestamp', datetime.now().strftime("%Y%m%d_%H%M%S"))
        output_filename = f"test_report_{timestamp}.html"
    
    results = combined_results.get('results', {})
    metadata = combined_results.get('metadata', {})
    
    # Format date/time
    test_date, test_time = format_datetime(metadata.get('test_date', ''))
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Rumble Rims Test Results Report</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Epilogue:wght@100;200;300;400;500;600;700;800;900&display=swap" rel="stylesheet">
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Epilogue', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #1E2D59 0%, #9E1B32 100%);
            padding: 20px;
            min-height: 100vh;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        
        .header {{
            background: #9E1B32;
            color: white;
            padding: 30px;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 20px;
        }}
        
        .header .logo {{
            height: 60px;
            width: auto;
            object-fit: contain;
        }}
        
        .header h1 {{
            font-size: 32px;
            margin: 0 0 10px 0;
        }}
        
        .header .subtitle {{
            font-size: 14px;
            opacity: 0.9;
        }}
        
        .metadata {{
            padding: 20px 30px;
            background: #f8f9fa;
            border-bottom: 2px solid #e9ecef;
            display: flex;
            justify-content: space-around;
            flex-wrap: wrap;
        }}
        
        .metadata-item {{
            text-align: center;
            margin: 5px;
        }}
        
        .metadata-label {{
            font-size: 11px;
            color: #6c757d;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        .metadata-value {{
            font-size: 16px;
            font-weight: bold;
            color: #212529;
            margin-top: 5px;
        }}
        
        .content {{
            padding: 30px;
        }}
        
        .section {{
            margin-bottom: 40px;
        }}
        
        .section-title {{
            font-size: 24px;
            color: #495057;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 3px solid #1E2D59;
        }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        
        .metric-card {{
            background: #f8f9fa;
            border-radius: 8px;
            padding: 20px;
            border-left: 4px solid #dee2e6;
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        
        .metric-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }}
        
        .metric-card.good {{
            border-left-color: #28a745;
        }}
        
        .metric-card.warning {{
            border-left-color: #ffc107;
        }}
        
        .metric-card.poor {{
            border-left-color: #dc3545;
        }}
        
        .metric-card.gray {{
            border-left-color: #6c757d;
        }}
        
        .metric-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }}
        
        .metric-name {{
            font-size: 16px;
            font-weight: 600;
            color: #212529;
        }}
        
        .metric-badge {{
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: bold;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .metric-badge.good {{
            background: #d4edda;
            color: #155724;
        }}
        
        .metric-badge.warning {{
            background: #fff3cd;
            color: #856404;
        }}
        
        .metric-badge.poor {{
            background: #f8d7da;
            color: #721c24;
        }}
        
        .metric-badge.gray {{
            background: #e2e3e5;
            color: #383d41;
        }}
        
        .metric-value {{
            font-size: 32px;
            font-weight: bold;
            color: #212529;
            margin: 10px 0;
        }}
        
        .metric-unit {{
            font-size: 14px;
            color: #6c757d;
            margin-left: 5px;
        }}
        
        .metric-scale {{
            margin-top: 15px;
            height: 8px;
            background: linear-gradient(to right, rgba(40, 167, 69, 0.3) 0%, rgba(40, 167, 69, 0.3) 33%, rgba(255, 193, 7, 0.3) 33%, rgba(255, 193, 7, 0.3) 66%, rgba(220, 53, 69, 0.3) 66%, rgba(220, 53, 69, 0.3) 100%);
            border-radius: 4px;
            position: relative;
            overflow: hidden;
        }}
        
        .metric-scale-bar {{
            height: 100%;
            border-radius: 4px;
            transition: width 0.5s ease;
            background: rgba(0, 0, 0, 0.7);
            box-shadow: 0 0 4px rgba(0, 0, 0, 0.3);
        }}
        
        .metric-thresholds {{
            display: flex;
            justify-content: space-between;
            margin-top: 8px;
            font-size: 10px;
            color: #6c757d;
        }}
        
        .metric-details {{
            margin-top: 10px;
            font-size: 12px;
            color: #6c757d;
        }}
        
        .score-dial-container {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 40px 20px;
            margin: 30px 0;
            gap: 30px;
        }}
        
        .score-dial-wrapper {{
            position: relative;
            text-align: center;
            flex: 0 0 auto;
        }}
        
        .score-dial {{
            width: 300px;
            height: 180px;
            max-width: 100%;
        }}
        
        .score-value {{
            font-size: 64px;
            font-weight: bold;
            color: #212529;
            flex: 0 0 auto;
            text-align: right;
            min-width: 100px;
        }}
        
        .score-message {{
            font-size: 18px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
            flex: 0 0 auto;
            text-align: left;
            min-width: 200px;
        }}
        
        .score-message.good {{
            color: #28a745;
        }}
        
        .score-message.warning {{
            color: #ffc107;
        }}
        
        .score-message.poor {{
            color: #dc3545;
        }}
        
        .eye-path-visualization {{
            margin: 20px 0;
            padding: 20px;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        .eye-path-legend {{
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-top: 15px;
            flex-wrap: wrap;
        }}
        
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 12px;
            color: #495057;
        }}
        
        .legend-color {{
            width: 16px;
            height: 16px;
            border-radius: 50%;
            border: 1px solid #dee2e6;
        }}
        
        .footer {{
            background: #f8f9fa;
            padding: 20px 30px;
            text-align: center;
            color: #6c757d;
            font-size: 12px;
        }}
        
        @media print {{
            body {{
                background: white;
                padding: 0;
            }}
            .container {{
                box-shadow: none;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <img src="ram_logo.png" alt="Rumble Rims Logo" class="logo">
            <div>
                <h1>Rumble Rims Test Results Report</h1>
                <div class="subtitle">Comprehensive Oculomotor Assessment</div>
            </div>
        </div>
        
        <div class="metadata">
            <div class="metadata-item">
                <div class="metadata-label">Test Date</div>
                <div class="metadata-value">{test_date if test_date else 'N/A'}</div>
            </div>
            <div class="metadata-item">
                <div class="metadata-label">Test Time</div>
                <div class="metadata-value">{test_time if test_time else 'N/A'}</div>
            </div>
            <div class="metadata-item">
                <div class="metadata-label">Monitor Resolution</div>
                <div class="metadata-value">{metadata.get('monitor_width', 'N/A')} × {metadata.get('monitor_height', 'N/A')}</div>
            </div>
        </div>
        
        <div class="content">
"""
    
    # Calculate overall score and add dial
    overall_score, total_weight = calculate_overall_score(results)
    
    if overall_score is not None:
        html_content += '<div class="section" style="text-align: center; padding: 20px 0;">'
        html_content += '<h2 class="section-title" style="border: none; margin-bottom: 10px;">Overall Assessment</h2>'
        html_content += create_score_dial_html(overall_score)
        html_content += '</div>'
    
    # Saccade Test Section
    if 'saccade' in results:
        html_content += '<div class="section">'
        html_content += '<h2 class="section-title">Saccade Test Results</h2>'
        
        # Try to get gaze path data from combined_results or load from saccade file
        gaze_path = None
        saccade_target_data = []
        
        # Check if gaze_path is in combined_results
        if 'saccade_data' in combined_results:
            gaze_path = combined_results['saccade_data'].get('gaze_path')
            saccade_target_data = combined_results['saccade_data'].get('normal_saccade_data', [])
        elif 'gaze_path' in combined_results:
            gaze_path = combined_results['gaze_path']
        elif 'normal_saccade_data' in combined_results:
            saccade_target_data = combined_results.get('normal_saccade_data', [])
            gaze_path = combined_results.get('gaze_path')
        
        # If not found, try to load from the most recent saccade results file
        if gaze_path is None:
            # Find all saccade results files
            saccade_files = glob.glob("saccade_results_*.json")
            
            if saccade_files:
                # Sort by modification time, most recent first
                saccade_files.sort(key=os.path.getmtime, reverse=True)
                
                # Try to load the most recent file
                try:
                    with open(saccade_files[0], 'r') as f:
                        saccade_data = json.load(f)
                        gaze_path = saccade_data.get('gaze_path')
                        if not saccade_target_data:
                            saccade_target_data = saccade_data.get('normal_saccade_data', [])
                except (FileNotFoundError, json.JSONDecodeError):
                    pass  # File not found or invalid, continue without visualization
        
        # Add eye path visualization if data is available
        if gaze_path:
            html_content += '<div style="margin-bottom: 30px;">'
            html_content += '<h3 style="font-size: 18px; color: #495057; margin-bottom: 15px;">Eye Movement Path</h3>'
            html_content += create_eye_path_visualization(
                gaze_path, 
                saccade_target_data,
                metadata.get('monitor_width', 1920),
                metadata.get('monitor_height', 1080)
            )
            html_content += '</div>'
        
        html_content += '<div class="metrics-grid">'
        
        normal = results['saccade'].get('normal', {})
        if normal:
            # Latency
            latency = normal.get('average_latency_ms', 0)
            color = get_color_for_value(latency, THRESHOLDS['saccade_latency_ms'], lower_is_better=True)
            html_content += create_metric_html('Saccade Latency', latency, 'ms', 
                                             THRESHOLDS['saccade_latency_ms'], color, 
                                             lower_is_better=True,
                                             details=f"Std: {normal.get('std_latency_ms', 0):.2f} ms")
            
            # Velocity
            velocity = normal.get('average_velocity_deg_per_ms', 0)
            color = get_color_for_value(velocity, THRESHOLDS['saccade_velocity_deg_per_ms'])
            html_content += create_metric_html('Saccade Velocity', velocity, 'deg/ms',
                                             THRESHOLDS['saccade_velocity_deg_per_ms'], color,
                                             details=f"Std: {normal.get('std_velocity_deg_per_ms', 0):.4f} deg/ms")
            
            # Accuracy
            accuracy = normal.get('average_accuracy_percent', 0)
            color = get_color_for_value(accuracy, THRESHOLDS['saccade_accuracy_percent'])
            html_content += create_metric_html('Saccade Accuracy', accuracy, '%',
                                             THRESHOLDS['saccade_accuracy_percent'], color,
                                             details=f"Valid: {normal.get('valid_saccades', 0)}/{normal.get('total_saccades', 0)}")
        
        anti = results['saccade'].get('antisaccade', {})
        if anti:
            error_rate = anti.get('error_rate_percent', 0)
            color = get_color_for_value(error_rate, THRESHOLDS['antisaccade_error_rate_percent'], lower_is_better=True)
            html_content += create_metric_html('Antisaccade Error Rate', error_rate, '%',
                                             THRESHOLDS['antisaccade_error_rate_percent'], color,
                                             lower_is_better=True,
                                             details=f"Errors: {anti.get('error_count', 0)}/{anti.get('total_trials', 0)}")
        
        html_content += '</div></div>'
    
    # Smooth Pursuit Section
    if 'smooth_pursuit' in results:
        html_content += '<div class="section">'
        html_content += '<h2 class="section-title">Smooth Pursuit Test Results</h2>'
        
        # Try to get gaze path data from the most recent smooth pursuit results file
        pursuit_gaze_path = None
        
        # Try to load from the most recent smooth pursuit results file
        pursuit_files = glob.glob("smooth_pursuit_results_*.json")
        
        if pursuit_files:
            # Sort by modification time, most recent first
            pursuit_files.sort(key=os.path.getmtime, reverse=True)
            
            # Try to load the most recent file
            try:
                with open(pursuit_files[0], 'r') as f:
                    pursuit_data = json.load(f)
                    pursuit_gaze_path = pursuit_data.get('gaze_path')
            except (FileNotFoundError, json.JSONDecodeError):
                pass  # File not found or invalid, continue without visualization
        
        # Add eye path visualization if data is available
        if pursuit_gaze_path:
            html_content += '<div style="margin-bottom: 30px;">'
            html_content += '<h3 style="font-size: 18px; color: #495057; margin-bottom: 15px;">Eye Movement Path</h3>'
            html_content += create_pursuit_path_visualization(
                pursuit_gaze_path,
                None,  # Target path not currently stored
                metadata.get('monitor_width', 1920),
                metadata.get('monitor_height', 1080)
            )
            html_content += '</div>'
        
        html_content += '<div class="metrics-grid">'
        
        pursuit = results['smooth_pursuit']
        
        gain = pursuit.get('average_gain', 0)
        color = get_color_for_value(gain, THRESHOLDS['smooth_pursuit_gain'])
        html_content += create_metric_html('Pursuit Gain', gain, '',
                                         THRESHOLDS['smooth_pursuit_gain'], color,
                                         details=f"H: {pursuit.get('average_gain_horizontal', 0):.3f}, V: {pursuit.get('average_gain_vertical', 0):.3f}")
        
        latency = pursuit.get('average_latency_ms', 0)
        if latency > 0:
            color = get_color_for_value(latency, THRESHOLDS['smooth_pursuit_latency_ms'], lower_is_better=True)
            html_content += create_metric_html('Pursuit Latency', latency, 'ms',
                                             THRESHOLDS['smooth_pursuit_latency_ms'], color,
                                             lower_is_better=True,
                                             details=f"Total measurements: {pursuit.get('total_measurements', 0)}")
        
        html_content += '</div></div>'
    
    # Fixed Point Stability Section
    if 'fixed_point' in results:
        html_content += '<div class="section">'
        html_content += '<h2 class="section-title">Fixed Point Stability Test Results</h2>'
        
        # Try to get gaze path data from the most recent fixed point results file
        fixed_gaze_path = None
        
        # Try to load from the most recent fixed point results file
        fixed_files = glob.glob("fixed_point_stability_results_*.json")
        
        if fixed_files:
            # Sort by modification time, most recent first
            fixed_files.sort(key=os.path.getmtime, reverse=True)
            
            # Try to load the most recent file
            try:
                with open(fixed_files[0], 'r') as f:
                    fixed_data = json.load(f)
                    fixed_gaze_path = fixed_data.get('gaze_path')
            except (FileNotFoundError, json.JSONDecodeError):
                pass  # File not found or invalid, continue without visualization
        
        # Add eye path visualization if data is available
        if fixed_gaze_path:
            html_content += '<div style="margin-bottom: 30px;">'
            html_content += '<h3 style="font-size: 18px; color: #495057; margin-bottom: 15px;">Eye Movement Path</h3>'
            # Center target is at screen center
            center_x = metadata.get('monitor_width', 1920) // 2
            center_y = metadata.get('monitor_height', 1080) // 2
            html_content += create_fixed_point_path_visualization(
                fixed_gaze_path,
                center_x,
                center_y,
                metadata.get('monitor_width', 1920),
                metadata.get('monitor_height', 1080)
            )
            html_content += '</div>'
        
        html_content += '<div class="metrics-grid">'
        
        fixed = results['fixed_point']
        deviation = fixed.get('average_deviation_degrees', 0)
        color = get_color_for_value(deviation, THRESHOLDS['fixed_point_deviation_degrees'], lower_is_better=True)
        html_content += create_metric_html('Gaze Deviation', deviation, 'degrees',
                                         THRESHOLDS['fixed_point_deviation_degrees'], color,
                                         lower_is_better=True,
                                         details=f"RMS: {fixed.get('rms_deviation_degrees', 0):.3f}°, Max: {fixed.get('max_deviation_degrees', 0):.3f}°")
        
        html_content += '</div></div>'
    
    # PLR Section
    if 'plr' in results:
        html_content += '<div class="section">'
        html_content += '<h2 class="section-title">Pupillary Light Reflex (PLR) Test Results</h2>'
        html_content += '<div class="metrics-grid">'
        
        plr_data = results['plr']
        
        latency = plr_data.get('plr_latency_ms')
        if latency is not None:
            color = get_color_for_value(latency, THRESHOLDS['plr_latency_ms'], lower_is_better=True)
            html_content += create_metric_html('PLR Latency', latency, 'ms',
                                             THRESHOLDS['plr_latency_ms'], color,
                                             lower_is_better=True)
        
        constriction = plr_data.get('constriction_amplitude_percent', 0)
        color = get_color_for_value(constriction, THRESHOLDS['plr_constriction_percent'])
        html_content += create_metric_html('Constriction Amplitude', constriction, '%',
                                         THRESHOLDS['plr_constriction_percent'], color,
                                         details=f"Baseline: {plr_data.get('baseline_diameter_pixels', 0):.1f} px")
        
        html_content += '</div></div>'
    
    # Blink Rate Section
    if 'blink_rate' in results:
        html_content += '<div class="section">'
        html_content += '<h2 class="section-title">Blink Rate</h2>'
        html_content += '<div class="metrics-grid">'
        
        blink = results['blink_rate']
        rate = blink.get('blinks_per_minute_60s', 0)
        color = get_color_for_value(rate, THRESHOLDS['blink_rate_per_min'])
        html_content += create_metric_html('Blink Rate', rate, 'blinks/min',
                                         THRESHOLDS['blink_rate_per_min'], color,
                                         details=f"Total blinks: {blink.get('total_blinks', 0)}")
        
        html_content += '</div></div>'
    
    html_content += """
        </div>
        
        <div class="footer">
            Report generated automatically by Eye Tracking Phygital System
        </div>
    </div>
</body>
</html>
"""
    
    # Write HTML file
    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"✓ HTML report saved to {output_filename}")
        return output_filename
    except Exception as e:
        print(f"✗ Error saving HTML report: {e}")
        import traceback
        traceback.print_exc()
        return None

