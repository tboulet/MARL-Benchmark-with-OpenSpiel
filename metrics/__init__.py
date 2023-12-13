from typing import Dict, List, Type
from metrics.base_metric import Metric
from metrics.episode_indexes import EpisodeIndexesMetric
from metrics.exploitability import ExploitabilityMetric
from metrics.vs_random import VsRandomMetric
from metrics.vs_human import VsHumanMetric
from metrics.inter_algo_faceoff import InterAlgoFaceoffMetric

metric_name_to_metric_class : Dict[str, Type[Metric]] = {
    "vs_random" : VsRandomMetric,
    "vs_human" : VsHumanMetric,
    "inter_algo_faceoff" : InterAlgoFaceoffMetric,
    "episode_idx" : EpisodeIndexesMetric,
    "exploitability" : ExploitabilityMetric,
}