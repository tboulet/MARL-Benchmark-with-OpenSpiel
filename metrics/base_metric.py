from abc import abstractmethod, ABC
from time import sleep
from typing import Callable, Dict, List, Type

from omegaconf import DictConfig
from open_spiel.python import rl_agent, rl_environment
import numpy as np



class Metric(ABC):
    """The base class for all metrics."""
    def __init__(self, config : DictConfig) -> None:
        """Initialize a metric object.

        Args:
            config (DictConfig): the config 
        """
        self.config = config

    @abstractmethod
    def evaluate(
        algo_name_to_grouped_agents : Dict[str, List[rl_agent.AbstractAgent]], 
        envs : Dict[str, rl_environment.Environment],
        episode_idx : int,
        ) -> Dict[str, float]:
        """The evaluation function.

        Args:
            algo_name_to_grouped_agents (Dict[str, List[rl_agent.AbstractAgent]]): the agents to evaluate, grouped by algorithm
            envs : Dict[str, rl_environment.Environment]: the environments to evaluate the agents in
            episode_idx (int): the current episode index
            
        Returns:
            Dict[str, float]: a dictionary of the metrics to return at each evaluation step, with the name of the metric as key and the value of the metric as value.
        """
        return {}
    
    

def evaluate_each_group_independently(
    algo_name_to_grouped_agents : Dict[str, List[rl_agent.AbstractAgent]],
    envs : Dict[str, rl_environment.Environment],
    evaluate_group_function : Callable[[List[rl_agent.AbstractAgent], rl_environment.Environment], float],
    ) -> Dict[str, float]:
    """A wrapper for evaluating each group of agents independently.

    Args:
        algo_name_to_grouped_agents (Dict[str, List[rl_agent.AbstractAgent]]): the agents to evaluate, grouped by algorithm
        envs (Dict[str, rl_environment.Environment]): the environments to evaluate the agents in
        evaluate_group_function (Callable[ [List[rl_agent.AbstractAgent], rl_environment.Environment], float]): the function used to evaluate each group of agents

    Returns:
        Dict[str, float]: a dictionary of the metrics to return at each evaluation step, with the name of the metric as key
    """
    metrics_dict = {}
    
    for algo_name, grouped_agents in algo_name_to_grouped_agents.items():
        metrics_dict_of_algo = evaluate_group_function(grouped_agents, envs[algo_name])
        for metric_name, metric_value in metrics_dict_of_algo.items():
            metrics_dict[f"{metric_name}/{algo_name}"] = metric_value
    return metrics_dict