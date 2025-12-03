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
    'saccade_latency_ms': {'good': 250, 'warning': 350},  # Lower is better
    'saccade_velocity_deg_per_ms': {'good': 0.2, 'warning': 0.15},  # Higher is better
    'saccade_accuracy_percent': {'good': 70, 'warning': 50},  # Higher is better
    'antisaccade_error_rate_percent': {'good': 20, 'warning': 40},  # Lower is better
    'smooth_pursuit_gain': {'good': 0.8, 'warning': 0.6},  # Higher is better
    'smooth_pursuit_latency_ms': {'good': 150, 'warning': 250},  # Lower is better
    'fixed_point_deviation_degrees': {'good': 1.0, 'warning': 2.0},  # Lower is better
    'plr_latency_ms': {'good': 300, 'warning': 400},  # Lower is better
    'plr_constriction_percent': {'good': 25, 'warning': 15},  # Higher is better
    'blink_rate_per_min': {'good': 20, 'warning': 10},  # Normal range
}

def get_color_for_value(value, thresholds, lower_is_better=False):
    """
    Return color class based on value and thresholds.
    
    Args:
        value: The metric value to evaluate
        thresholds: Dictionary with 'good' and 'warning' keys
        lower_is_better: If True, lower values are better (e.g., latency, error rate)
    
    Returns:
        'good', 'warning', 'poor', or 'gray' (if value is None)
    """
    if value is None:
        return 'gray'
    
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
        thresholds: Dictionary with 'good' and 'warning' keys
        lower_is_better: Whether lower values are better
    
    Returns:
        Score from 0-100, or None if value is None
    """
    if value is None:
        return None
    
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
    
    return overall_score, total_weight

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
    # Dial: 0° (right) to 180° (left)
    # Score 0 → 180° (left), Score 100 → 0° (right)
    # Needle starts pointing up (which is -90° from horizontal), so we rotate by (angle - 90) to point along the arc
    angle = (score / 100) * 180
    needle_rotation = angle - 90
    
    # Determine color and message based on score
    if score >= 67:  # Good range
        color = '#28a745'
        message = 'No concussion detected'
        label_class = 'good'
    elif score >= 33:  # Warning range
        color = '#ffc107'
        message = 'Potential concussion detected'
        label_class = 'warning'
    else:  # Poor range
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
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
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
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 32px;
            margin-bottom: 10px;
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
            border-bottom: 3px solid #667eea;
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
            <h1>Rumble Rims Test Results Report</h1>
            <div class="subtitle">Comprehensive Oculomotor Assessment</div>
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

