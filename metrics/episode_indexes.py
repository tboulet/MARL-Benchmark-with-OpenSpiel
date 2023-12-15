from abc import abstractmethod, ABC
from time import sleep
from omegaconf import DictConfig
from typing import Dict, List, Type, Union
import numpy as np

from open_spiel.python import rl_agent, rl_environment
from open_spiel.python.algorithms import random_agent

from metrics.base_metric import Metric, evaluate_each_group_independently



class EpisodeIndexesMetric(Metric):
    """Simply log the number of training episodes already done."""
    def __init__(self, 
            config : DictConfig,
            eval_frequency : int = 1000,
            ) -> None:
        """Initialize the EpisodeIndexeMetric, that simply regularly log the number of training episodes already done.

        Args:
            config (DictConfig): the configuration of the run
            eval_frequency (int, optional): the frequency of the logging. Defaults to 1000.
        """
        self.eval_frequency = eval_frequency
        super().__init__(config = config)

    def evaluate(self, 
            group_names_to_grouped_agents : Dict[str, List[rl_agent.AbstractAgent]],
            group_names_to_envs : List[rl_environment.Environment],
            episode_idx : int,
            ) -> Dict[str, float]:
        if not episode_idx % self.eval_frequency == 0:
            return {}
        else:
            print(f"[Episode Indexes Metric] Number of training episodes done: {episode_idx}")
            metrics_dict = {"episode_idx": episode_idx}
            return metrics_dict

