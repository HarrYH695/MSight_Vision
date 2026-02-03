from typing import Dict, List, Tuple
from .base import DetectionResult2D
from geopy.distance import geodesic
from .utils import detection_to_roaduser_point
from msight_base import RoadUserPoint
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Point, Polygon
import numpy as np

class FuserBase:
    def fuse(self, results: Dict[str, DetectionResult2D]) -> List[RoadUserPoint]:
        """
        Fuses the data from different sources into a single output.
        :param data: The input data to be fused.
        :return: The fused output.
        """
        raise NotImplementedError("FuserBase is an abstract class and cannot be instantiated directly.")
    
## This is a simple example of a fuser that combines the results from different cameras, which works at the roundabout of State and Ellsworth in Smart Intersection Project.
class StateEllsworthFuser(FuserBase):
    '''
    This is a simple example of a fuser that combines the results from different cameras, which works at the roundabouot of State and Ellsworth.
    '''
    def __init__(self):
        self.lat1 = 42.229379
        self.lon1 = -83.739003
        self.lat2 = 42.229444
        self.lon2 = -83.739013

    def fuse(self, detection_buffer: Dict[str, DetectionResult2D]) -> List[RoadUserPoint]:
        fused_vehicle_list = []

        vehicle_list = detection_buffer['gs_State_Ellsworth_NW'].object_list
        for v in vehicle_list: # cam_ne
            if v.lat > self.lat1 and v.lon > self.lon1:
                fused_vehicle_list.append(detection_to_roaduser_point(v, 'gs_State_Ellsworth_NW'))

        vehicle_list = detection_buffer['gs_State_Ellsworth_NE'].object_list
        for v in vehicle_list: # cam_nw
            if v.lat > self.lat2 and v.lon < self.lon2:
                fused_vehicle_list.append(detection_to_roaduser_point(v, 'gs_State_Ellsworth_NE'))

        vehicle_list = detection_buffer['gs_State_Ellsworth_SE'].object_list
        for v in vehicle_list: # cam_se
            if v.lat < self.lat1 and v.lon > self.lon1:
                fused_vehicle_list.append(detection_to_roaduser_point(v, 'gs_State_Ellsworth_SE'))

        vehicle_list = detection_buffer['gs_State_Ellsworth_SW'].object_list
        for v in vehicle_list: # cam_sw
            if v.lat < self.lat2 and v.lon < self.lon2:
                fused_vehicle_list.append(detection_to_roaduser_point(v, 'gs_State_Ellsworth_SW'))
        return fused_vehicle_list

class HungarianFuser(FuserBase):
    """
    A fuser that matches detections from multiple sensors based on spatial proximity
    using Hungarian algorithm and fuses their locations using weighted averaging.
    """
    def __init__(self, coverage_zones: dict, sensor_locations: dict = None, distance_threshold: float = 5.0):
        """
        Initialize the HungarianFuser.
        :param coverage_zones: dict mapping sensor_id to a polygon defining the sensor's coverage zone.
                               Each polygon is a list of (lat, lon) tuples forming a closed polygon.
                               Example: [(lat1, lon1), (lat2, lon2), (lat3, lon3), ...]
        :param sensor_locations: dict mapping sensor_id to (lat, lon) tuple of the sensor's location.
                                 If provided, weights are computed as 1/distance_to_sensor^2.
                                 If not provided, bounding box area is used as weight.
        :param distance_threshold: maximum distance (in meters) to consider two detections as the same object.
        """
        self.coverage_zones = coverage_zones
        self.sensor_locations = sensor_locations
        self.distance_threshold = distance_threshold
        self.sensor_list = list(coverage_zones.keys())
        
        # Pre-compute Shapely Polygon objects for efficient point-in-polygon checks
        self._coverage_polygons = {
            sensor_id: Polygon(polygon) if polygon else None
            for sensor_id, polygon in coverage_zones.items()
        }

    def _is_in_coverage(self, detected_object, sensor_id: str) -> bool:
        """
        Check if a detected object is within the sensor's coverage zone.
        :param detected_object: DetectedObject2D instance
        :param sensor_id: sensor identifier
        :return: True if in coverage zone, False otherwise
        """
        polygon = self._coverage_polygons.get(sensor_id)
        if polygon is None:
            return True  # No coverage filter defined, include all
        point = Point(detected_object.lat, detected_object.lon)
        return polygon.contains(point)

    def _compute_weight(self, detected_object, sensor_id: str) -> float:
        """
        Compute the weight for a detected object.
        If sensor location is available, use 1/distance_to_sensor^2.
        Otherwise, use the bounding box area.
        :param detected_object: DetectedObject2D instance
        :param sensor_id: sensor identifier
        :return: weight value
        """
        if self.sensor_locations is not None and sensor_id in self.sensor_locations:
            sensor_lat, sensor_lon = self.sensor_locations[sensor_id]
            # Compute geodesic distance to sensor in meters
            dist = geodesic((detected_object.lat, detected_object.lon), (sensor_lat, sensor_lon)).meters
            dist_sq = dist ** 2
            if dist_sq < 1e-10:
                dist_sq = 1e-10  # Avoid division by zero
            return 1.0 / dist_sq
        else:
            # Use bounding box area as weight
            # box is [x1, y1, x2, y2]
            box = detected_object.box
            width = box[2] - box[0]
            height = box[3] - box[1]
            area = width * height
            return area if area > 0 else 1.0

    def _compute_distance_to_group(self, detected_object, group: dict) -> float:
        """
        Compute the geodesic distance between a detected object and a group's weighted location.
        :param detected_object: DetectedObject2D instance
        :param group: group dict containing 'weighted_lat' and 'weighted_lon'
        :return: distance in meters
        """
        return geodesic(
            (detected_object.lat, detected_object.lon),
            (group['weighted_lat'], group['weighted_lon'])
        ).meters

    def _filter_detections_by_sensor(self, detection_buffer: Dict[str, DetectionResult2D]) -> Dict[str, List]:
        """
        Filter detections by coverage zone and organize by sensor.
        :param detection_buffer: dict mapping sensor_id to DetectionResult2D
        :return: dict mapping sensor_id to list of valid DetectedObject2D instances
        """
        detections_by_sensor = {}
        for sensor_id in self.sensor_list:
            if sensor_id not in detection_buffer:
                continue
            detection_result = detection_buffer[sensor_id]
            valid_detections = []
            for detected_object in detection_result.object_list:
                # Skip objects without valid lat/lon
                if detected_object.lat is None or detected_object.lon is None:
                    continue
                # Filter by coverage zone
                if self._is_in_coverage(detected_object, sensor_id):
                    valid_detections.append(detected_object)
            if valid_detections:
                detections_by_sensor[sensor_id] = valid_detections
        return detections_by_sensor

    def _create_group_from_detection(self, detected_object, sensor_id: str) -> dict:
        """
        Create a new group from a single detection.
        :param detected_object: DetectedObject2D instance
        :param sensor_id: sensor identifier
        :return: group dict
        """
        weight = self._compute_weight(detected_object, sensor_id)
        return {
            'weighted_lat': detected_object.lat,
            'weighted_lon': detected_object.lon,
            'total_weight': weight,
            'weighted_lat_sum': detected_object.lat * weight,
            'weighted_lon_sum': detected_object.lon * weight,
            'max_confidence': detected_object.score,
            'class_id_counts': {detected_object.class_id: 1},
            'sensor_data': {sensor_id: detected_object},
        }

    def _add_detection_to_group(self, group: dict, detected_object, sensor_id: str) -> None:
        """
        Add a detection to an existing group and update weighted location.
        :param group: group dict to update
        :param detected_object: DetectedObject2D instance
        :param sensor_id: sensor identifier
        """
        weight = self._compute_weight(detected_object, sensor_id)
        
        # Update weighted sums
        group['weighted_lat_sum'] += detected_object.lat * weight
        group['weighted_lon_sum'] += detected_object.lon * weight
        group['total_weight'] += weight
        
        # Update weighted location
        group['weighted_lat'] = group['weighted_lat_sum'] / group['total_weight']
        group['weighted_lon'] = group['weighted_lon_sum'] / group['total_weight']
        
        # Update confidence
        if detected_object.score > group['max_confidence']:
            group['max_confidence'] = detected_object.score
        
        # Update class_id counts
        class_id = detected_object.class_id
        group['class_id_counts'][class_id] = group['class_id_counts'].get(class_id, 0) + 1
        
        # Store sensor data (keep the detected object directly)
        group['sensor_data'][sensor_id] = detected_object

    def _hungarian_match(self, groups: List[dict], detections: List, sensor_id: str) -> Tuple[List[dict], List]:
        """
        Use Hungarian algorithm to match detections to existing groups.
        :param groups: list of existing group dicts
        :param detections: list of DetectedObject2D instances from a single sensor
        :param sensor_id: sensor identifier
        :return: (updated groups, unmatched detections)
        """
        if not groups or not detections:
            return groups, detections
        
        n_groups = len(groups)
        n_detections = len(detections)
        
        # Build cost matrix (distance between each detection and each group)
        cost_matrix = np.zeros((n_detections, n_groups))
        for i, det in enumerate(detections):
            for j, group in enumerate(groups):
                dist = self._compute_distance_to_group(det, group)
                cost_matrix[i, j] = dist
        
        # Apply Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        matched_detection_indices = set()
        matched_group_indices = set()
        
        # Process matches
        for det_idx, group_idx in zip(row_ind, col_ind):
            dist = cost_matrix[det_idx, group_idx]
            if dist <= self.distance_threshold:
                # Valid match - add detection to group
                self._add_detection_to_group(groups[group_idx], detections[det_idx], sensor_id)
                matched_detection_indices.add(det_idx)
                matched_group_indices.add(group_idx)
        
        # Collect unmatched detections
        unmatched_detections = [
            detections[i] for i in range(n_detections) if i not in matched_detection_indices
        ]
        
        return groups, unmatched_detections

    def _group_to_road_user_point(self, group: dict) -> RoadUserPoint:
        """
        Convert a group to a RoadUserPoint.
        :param group: group dict
        :return: RoadUserPoint instance
        """
        # Determine the most common class_id
        most_common_class = max(group['class_id_counts'], key=group['class_id_counts'].get)
        
        # Convert sensor_data to dict format at the final stage
        sensor_data_dict = {
            sensor_id: det_obj.to_dict() 
            for sensor_id, det_obj in group['sensor_data'].items()
        }
        
        # Create the fused RoadUserPoint
        road_user_point = RoadUserPoint(
            x=group['weighted_lat'],
            y=group['weighted_lon'],
            category=most_common_class,
            confidence=group['max_confidence'],
        )
        road_user_point.sensor_data = sensor_data_dict
        
        return road_user_point

    def fuse(self, detection_buffer: Dict[str, DetectionResult2D]) -> List[RoadUserPoint]:
        """
        Fuse detections from multiple sensors using Hungarian matching.
        
        Algorithm:
        1. For the first sensor, create a group for each detection (single object groups)
        2. For each subsequent sensor:
           a. Use Hungarian algorithm to match detections to existing groups
           b. For matched pairs within distance_threshold, add detection to group and update weighted location
           c. For unmatched detections, create new single-object groups
        3. Convert all groups to RoadUserPoints
        
        :param detection_buffer: dict mapping sensor_id to DetectionResult2D
        :return: list of fused RoadUserPoint instances
        """
        # Step 1: Filter and organize detections by sensor
        detections_by_sensor = self._filter_detections_by_sensor(detection_buffer)
        
        if not detections_by_sensor:
            return []
        
        # Get list of sensors that have detections (preserve order from sensor_list)
        active_sensors = [s for s in self.sensor_list if s in detections_by_sensor]
        
        if not active_sensors:
            return []
        
        # Step 2: Initialize groups with first sensor's detections
        first_sensor = active_sensors[0]
        groups = []
        for det in detections_by_sensor[first_sensor]:
            group = self._create_group_from_detection(det, first_sensor)
            groups.append(group)
        
        # Step 3: Process remaining sensors
        for sensor_id in active_sensors[1:]:
            detections = detections_by_sensor[sensor_id]
            
            # Hungarian matching against existing groups
            groups, unmatched_detections = self._hungarian_match(groups, detections, sensor_id)
            
            # Create new groups for unmatched detections
            for det in unmatched_detections:
                group = self._create_group_from_detection(det, sensor_id)
                groups.append(group)
        
        # Step 4: Convert groups to RoadUserPoints
        fused_results = []
        for group in groups:
            road_user_point = self._group_to_road_user_point(group)
            fused_results.append(road_user_point)
        
        return fused_results
            