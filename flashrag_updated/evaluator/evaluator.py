import os
import json
import os
import numpy as np
from flashrag.evaluator.metrics import BaseMetric
from copy import deepcopy
from collections import deque
import re
import torch

class Evaluator:
    """Evaluator is used to summarize the results of all metrics."""

    def __init__(self, config):
        self.config = config
        self.save_dir = config['save_dir']

        self.save_metric_flag = config['save_metric_score']
        self.save_data_flag = config['save_intermediate_data']
        self.metrics = [metric.lower() for metric in self.config['metrics']]

        self.avaliable_metrics = self._collect_metrics()

        self.metric_class = {}
        for metric in self.metrics:
            if metric in self.avaliable_metrics:
                self.metric_class[metric] = self.avaliable_metrics[metric](self.config)
            else:
                print(f"{metric} has not been implemented!")
                raise NotImplementedError

    def _collect_metrics(self):
        """Collect all classes based on ```BaseMetric```."""

        def find_descendants(base_class, subclasses=None):
            if subclasses is None:
                subclasses = set()

            direct_subclasses = base_class.__subclasses__()
            for subclass in direct_subclasses:
                if subclass not in subclasses:
                    subclasses.add(subclass)
                    find_descendants(subclass, subclasses)
            return subclasses

        avaliable_metrics = {}
        for cls in find_descendants(BaseMetric):
            metric_name = cls.metric_name
            avaliable_metrics[metric_name] = cls
        return avaliable_metrics

    def evaluate(self, data):
        """Calculate all metric indicators and summarize them."""

        result_dict = {}
        for metric in self.metrics:
            try:
                metric_result, metric_scores = self.metric_class[metric].calculate_metric(data)
                result_dict.update(metric_result)

                for metric_score, item in zip(metric_scores, data):
                    item.update_evaluation_score(metric, metric_score)
            except Exception as e:
                print(f'Error in {metric}!')
                print(e)
                continue

        if self.save_metric_flag:
            self.save_metric_score(result_dict)

        if self.save_data_flag:
            pass
            # self.save_data(data)


        return result_dict

    def save_metric_score(self, result_dict):
        file_name = "metric_score.txt"
        save_path = os.path.join(self.save_dir, file_name)
        with open(save_path, "w", encoding='utf-8') as f:
            for k,v in result_dict.items():
                f.write(f"{k}: {v}\n")


    def save_data(self, data):
        """Save the evaluated data, including the raw data and the score of each data 
        sample on each metric."""

        file_name = "intermediate_data.json"
        save_path = os.path.join(self.save_dir, file_name)

        data.save(save_path)

    def save_data2(self, data):
        """Save the evaluated data, including the raw data and the score of each data 
        sample on each metric."""

        def convert_to_serializable(obj):
            if isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(i) for i in obj]
            elif hasattr(obj, "__dict__"):
                return {k: convert_to_serializable(v) for k, v in obj.__dict__.items()}
            elif hasattr(obj, "__iter__") and not isinstance(obj, str):
                return [convert_to_serializable(i) for i in obj]
            elif isinstance(obj, re.Pattern):
                return obj.pattern
            elif isinstance(obj, torch.device):
                return str(obj)
            return obj

        # Ensure all data is JSON serializable by deeply copying and converting it
        serializable_data = convert_to_serializable(data)

        # Save path setup
        file_name = "intermediate_data.json"
        save_path = os.path.join(self.save_dir, file_name)

        # Manually write the JSON serializable data to file
        with open(save_path, 'w') as file:
            json.dump(serializable_data, file, indent=2)
