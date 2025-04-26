import pandas as pd
import numpy as np
from geopy.distance import geodesic
from scipy.stats import skew, kurtosis

def extract_trajectory_features(df):
    # Initialize a list to store each trajectory's computed features
    features = []

    # Process each unique trajectory
    for tid, group in df.groupby('tid'):
        group = group.sort_values('time').reset_index(drop=True)
        
        ####################### Distance-based geometry features #######################
        # Calculate total trajectory length and segment-wise distances
        segment_lengths = [
            geodesic((group.iloc[i]['lat'], group.iloc[i]['lon']),
                     (group.iloc[i+1]['lat'], group.iloc[i+1]['lon'])).meters
            for i in range(len(group) - 1)
        ]
        
        # Initialize a dictionary to store features for the current trajectory
        trajectory_features = {'tid': tid}
        
        # Calculate curvature features
        signatures = []
        for level in range(1, 6):
            num_segments = level
            segment_size = len(group) // num_segments
            
            for j in range(num_segments):
                start = j * segment_size
                end = (j + 1) * segment_size if (j + 1) * segment_size < len(group) else len(group) - 1
                # print(f"start: {start}, end: {end}")
                segment_distance = geodesic((group.iloc[start]['lat'], group.iloc[start]['lon']),
                                            (group.iloc[end]['lat'], group.iloc[end]['lon'])).meters
                level_signature = segment_distance / (sum(segment_lengths[start:end]) if sum(segment_lengths[start:end]) > 0 else 1)
                signatures.append(level_signature)            
        
        column_names = [
            "distance_geometry_1_1", "distance_geometry_2_1", "distance_geometry_2_2",
            "distance_geometry_3_1", "distance_geometry_3_2", "distance_geometry_3_3",
            "distance_geometry_4_1", "distance_geometry_4_2", "distance_geometry_4_3", "distance_geometry_4_4",
            "distance_geometry_5_1", "distance_geometry_5_2", "distance_geometry_5_3", "distance_geometry_5_4", "distance_geometry_5_5"
        ]
        
        for k, name in enumerate(column_names):
            trajectory_features[name] = signatures[k] if k < len(signatures) else np.nan
        
        ####################### Indentation-based geometry features ####################### 
        # Calculate angles between consecutive segments
        angles = []
        for i in range(1, len(group) - 1):
            # Points for two consecutive segments
            p1 = (group.iloc[i - 1]['lat'], group.iloc[i - 1]['lon'])
            p2 = (group.iloc[i]['lat'], group.iloc[i]['lon'])
            p3 = (group.iloc[i + 1]['lat'], group.iloc[i + 1]['lon'])
                    
            # Calculate vectors
            v1 = (p2[0] - p1[0], p2[1] - p1[1])
            v2 = (p3[0] - p2[0], p3[1] - p2[1])
                    
            # Calculate angle between vectors
            dot_product = v1[0] * v2[0] + v1[1] * v2[1]
            norm_v1 = np.sqrt(v1[0] ** 2 + v1[1] ** 2)
            norm_v2 = np.sqrt(v2[0] ** 2 + v2[1] ** 2)
                    
            # Avoid division by zero in case of stationary points
            if norm_v1 > 0 and norm_v2 > 0:
                cos_theta = dot_product / (norm_v1 * norm_v2)
                angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # Angle in radians
                angles.append(np.degrees(angle))  # Convert to degrees
                
        # Compute statistical features if angles are available
        if angles:
            angles = np.array(angles)
            trajectory_features.update({
                'angles_0s': np.sum(angles == 0),
                'angles_mean': np.mean(angles),
                'angles_meanse': np.std(angles) / np.sqrt(len(angles)),
                'angles_quant_min': np.min(angles),
                'angles_quant_05': np.percentile(angles, 5),
                'angles_quant_10': np.percentile(angles, 10),
                'angles_quant_25': np.percentile(angles, 25),
                'angles_quant_median': np.median(angles),
                'angles_quant_75': np.percentile(angles, 75),
                'angles_quant_90': np.percentile(angles, 90),
                'angles_quant_95': np.percentile(angles, 95),
                'angles_quant_max': np.max(angles),
                'angles_range': np.ptp(angles),
                'angles_sd': np.std(angles),
                'angles_vcoef': np.std(angles) / np.mean(angles) if np.mean(angles) != 0 else np.nan,
                'angles_mad': np.median(np.abs(angles - np.median(angles))),
                'angles_iqr': np.percentile(angles, 75) - np.percentile(angles, 25),
                'angles_skew': skew(angles) if np.std(angles) > 1e-6 else 0.0,
                'angles_kurt': kurtosis(angles) if np.std(angles) > 1e-6 else 0.0

            })
        
        ####################### Speed and Acceleration Features #######################
        # Calculate speeds (m/s) and accelerations (m/s^2)
        speeds = []
        accelerations = []
        for i in range(len(group) - 2):
            # Points for speed and acceleration calculations
            p1 = (group.iloc[i]['lat'], group.iloc[i]['lon'])
            p2 = (group.iloc[i + 1]['lat'], group.iloc[i + 1]['lon'])
            p3 = (group.iloc[i + 2]['lat'], group.iloc[i + 2]['lon'])
                
            # Distances between points
            distance_1 = geodesic(p1, p2).meters
            distance_2 = geodesic(p2, p3).meters
                
            # Time differences between points
            time_diff_1 = (group.iloc[i + 1]['time'] - group.iloc[i]['time']).total_seconds()
            time_diff_2 = (group.iloc[i + 2]['time'] - group.iloc[i + 1]['time']).total_seconds()
                
            # Calculate speed if time_diff is positive
            if time_diff_1 > 0:
                speed_1 = distance_1 / time_diff_1
                speeds.append(speed_1)
                    
                if time_diff_2 > 0:
                    speed_2 = distance_2 / time_diff_2
                    # Calculate acceleration
                    accel_time_diff = (group.iloc[i + 2]['time'] - group.iloc[i]['time']).total_seconds()
                    if accel_time_diff > 0:
                        acceleration = (speed_2 - speed_1) / accel_time_diff
                        accelerations.append(acceleration)
        
        def compute_stats(data, prefix):
            if len(data) == 0:
                return {f"{prefix}_{stat}": np.nan for stat in [
            '0s', 'mean', 'meanse', 'quant_min', 'quant_05', 'quant_10', 'quant_25', 
            'quant_median', 'quant_75', 'quant_90', 'quant_95', 'quant_max', 'range', 
            'sd', 'vcoef', 'mad', 'iqr', 'skew', 'kurt']}

            data = np.array(data)
            std_dev = np.std(data)

            return {
                f"{prefix}_0s": np.sum(data == 0),
                f"{prefix}_mean": np.mean(data),
                f"{prefix}_meanse": std_dev / np.sqrt(len(data)),
                f"{prefix}_quant_min": np.min(data),
                f"{prefix}_quant_05": np.percentile(data, 5),
                f"{prefix}_quant_10": np.percentile(data, 10),
                f"{prefix}_quant_25": np.percentile(data, 25),
                f"{prefix}_quant_median": np.median(data),
                f"{prefix}_quant_75": np.percentile(data, 75),
                f"{prefix}_quant_90": np.percentile(data, 90),
                f"{prefix}_quant_95": np.percentile(data, 95),
                f"{prefix}_quant_max": np.max(data),
                f"{prefix}_range": np.ptp(data),
                f"{prefix}_sd": std_dev,
                f"{prefix}_vcoef": std_dev / np.mean(data) if np.mean(data) != 0 else np.nan,
                f"{prefix}_mad": np.median(np.abs(data - np.median(data))),
                f"{prefix}_iqr": np.percentile(data, 75) - np.percentile(data, 25),
                f"{prefix}_skew": skew(data) if std_dev > 1e-6 else 0.0,
                f"{prefix}_kurt": kurtosis(data) if std_dev > 1e-6 else 0.0,
     }


        # Compute and store statistics for speed and acceleration
        speed_stats = compute_stats(speeds, "speed")
        acceleration_stats = compute_stats(accelerations, "acceleration")
            
        # Combine all features for this trajectory
        trajectory_features.update(speed_stats)
        trajectory_features.update(acceleration_stats)
        # Add the label for the current trajectory
        trajectory_features['label'] = group['label'].iloc[0]
        
        # Append the computed features for the current trajectory
        features.append(trajectory_features)
        
    # Convert features list to DataFrame
    features_df = pd.DataFrame(features)
    
    return features_df

