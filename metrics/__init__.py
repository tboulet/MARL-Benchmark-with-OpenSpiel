from typing import Dict, List, Type
from metrics.base_metric import Metric
from metrics.vs_random import VsRandomMetric
from metrics.vs_human import VsHumanMetric

metric_name_to_metric_class : Dict[str, Type[Metric]] = {
    "vs_random" : VsRandomMetric,
    "vs_human" : VsHumanMetric,
}