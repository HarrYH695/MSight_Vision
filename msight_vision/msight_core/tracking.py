from msight_core.nodes import DataProcessingNode, NodeConfig
from msight_core.data import RoadUserListData
from .. import SortTracker
from pathlib import Path
import yaml
import time


class SortTrackerNode(DataProcessingNode):
    default_configs = NodeConfig(
        publish_topic_data_type=RoadUserListData
    )
    def __init__(self, configs, tracking_configs_path):
        super().__init__(configs)
        self.config_file_path = Path(tracking_configs_path)
        with open(self.config_file_path, "r") as f:
            self.tracking_configs = yaml.safe_load(f)
        self.max_age = self.tracking_configs.get("max_age", 3)
        self.min_hits = self.tracking_configs.get("min_hits", 1)
        self.iou_threshold = self.tracking_configs.get("iou_threshold", 0.01)
        self.use_filtered_position = self.tracking_configs.get("use_filtered_position", False)
        self.output_predicted = self.tracking_configs.get("output_predicted", False)
        self.tracker = SortTracker(
            max_age=self.max_age, 
            min_hits=self.min_hits, 
            iou_threshold=self.iou_threshold, 
            use_filtered_position=self.use_filtered_position, 
            output_predicted=self.output_predicted
        )
        

    def process(self, data: RoadUserListData) -> RoadUserListData:
        self.logger.info(f"Processing road user list data from sensor: {data.sensor_name}")
        start = time.time()
        road_user_list = data.road_user_list
        # print(road_user_list)
        tracking_result = self.tracker.track(road_user_list)
        data.road_user_list = tracking_result
        self.logger.info(f"Tracking completed in {time.time() - start:.2f} seconds for sensor: {data.sensor_name}")
        return data

    