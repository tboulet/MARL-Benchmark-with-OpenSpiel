from abc import abstractmethod, ABC
from time import sleep
from omegaconf import DictConfig
from typing import Dict, List, Type
import numpy as np

from open_spiel.python import rl_agent, rl_environment
from open_spiel.python.algorithms import random_agent

from metrics.base_metric import Metric, evaluate_each_group_independently



class VsRandomMetric(Metric):
    """A metric that regularly evaluates the agents against random agents."""
    
    def __init__(self, 
            config : DictConfig,
            eval_frequency: int, 
            n_episodes_evaluation: int,
            ) -> None:
        self.eval_frequency = eval_frequency
        self.n_episodes_evaluation = n_episodes_evaluation
        super().__init__(config = config)


    def evaluate(self, 
            algo_name_to_grouped_agents : Dict[str, List[rl_agent.AbstractAgent]],
            envs : List[rl_environment.Environment],
            episode_idx : int,
            ) -> Dict[str, float]:
        """Evaluate each agents against random agents.
        
        First version : 
        - only focus on the first agent (we could do average, min and max, or all) (TODO)
        - assume this agent starts to play every game (we could randomize the order) (TODO)

        Args:
            algo_name_to_grouped_agents (Dict[str, List[rl_agent.AbstractAgent]]): the agents to evaluate, grouped by algorithm
            envs (List[rl_environment.Environment]): the environments to evaluate the agents in
            episode_idx (int): the current episode index
            
        Returns:
            Dict[str, float]: a dictionary of the metrics to return at each evaluation step, with
        """

        if not episode_idx % self.eval_frequency == 0:
            return {}
        
        def evaluate_group_function_vs_random(grouped_agents : List[rl_agent.AbstractAgent], env : rl_environment.Environment) -> Dict[str, float]:
            """This function evaluates a group of agents against random agents."""
            num_players = env.num_players
            num_actions = env.action_spec()["num_actions"]
            evaluated_agents = [grouped_agents[0]] +[random_agent.RandomAgent(player_id, num_actions, "Random Agent") for player_id in range(1, num_players)]

            G_0s = []   # [G_0^ep | ep in [0, n_episodes_evaluation-1]]
            are_victories = []   # [1(G_0^ep > G_0_adv^ep) | ep in [0, n_episodes_evaluation-1]]
            
            for episode in range(self.n_episodes_evaluation):
                time_step = env.reset()
                G_0 = 0
                G_0_adv = 0
                while not time_step.last():
                    player_id = time_step.observations["current_player"]
                    agent_output = evaluated_agents[player_id].step(time_step, is_evaluation=True)   # Note the evaluation flag. A Q-learner will set epsilon=0 here.
                    time_step = env.step([agent_output.action])
                    G_0 += time_step.rewards[0]
                    G_0_adv += time_step.rewards[1]
                for agent in grouped_agents:
                    agent.step(time_step)
                G_0s.append(G_0)
                are_victories.append(int(G_0 > G_0_adv))
                
            mean_G_0 = np.mean(G_0s)
            std_G_0 = np.std(G_0s)
            victory_rate = np.mean(are_victories)
            
            return {
                    "episode_idx": episode_idx,
                    "mean_reward_vs_random": mean_G_0,
                    "std_reward_vs_random": std_G_0,
                    "victory_percentage_vs_random": victory_rate,
                }


        return evaluate_each_group_independently(
            algo_name_to_grouped_agents=algo_name_to_grouped_agents,
            envs=envs,
            evaluate_group_function=evaluate_group_function_vs_random,
        )

