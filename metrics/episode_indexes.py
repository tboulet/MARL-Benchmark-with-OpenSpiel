from abc import abstractmethod, ABC
from time import sleep
from omegaconf import DictConfig
from typing import Dict, List, Type, Union
import numpy as np

from open_spiel.python import rl_agent, rl_environment
from open_spiel.python.algorithms import random_agent

from metrics.base_metric import Metric, evaluate_each_group_independently



class EpisodeIndexesMetric(Metric):
    
    def __init__(self, 
            config : DictConfig,
            ) -> None:
        super().__init__(config = config)

    def evaluate(self, 
            group_names_to_grouped_agents : Dict[str, List[rl_agent.AbstractAgent]],
            group_names_to_envs : List[rl_environment.Environment],
            episode_idx : int,
            ) -> Dict[str, float]:

        metrics_dict = {"episode_idx": episode_idx}
        return metrics_dict

