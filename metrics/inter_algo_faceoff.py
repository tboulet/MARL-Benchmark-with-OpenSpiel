from abc import abstractmethod, ABC
from time import sleep
from omegaconf import DictConfig
from typing import Dict, List, Type
import numpy as np

from open_spiel.python import rl_agent, rl_environment
from open_spiel.python.algorithms import random_agent

from metrics.base_metric import Metric



class InterAlgoFaceoffMetric(Metric):
    """The metric that compares the agents to each other."""
    
    def __init__(self, 
            config : DictConfig,
            eval_frequency: int, 
            n_episodes_evaluation: int,
            game_name : str,
            game_config : Dict,
            ) -> None:
        self.eval_frequency = eval_frequency
        self.n_episodes_evaluation = n_episodes_evaluation
        self.game_name = game_name
        self.game_config = game_config
        super().__init__(config = config)
        
    
    def evaluate(self, 
            algo_name_to_grouped_agents : Dict[str, List[rl_agent.AbstractAgent]],
            envs : List[rl_environment.Environment],
            episode_idx : int,
            ) -> Dict[str, float]:
        """Evaluate the different algorithms against each other.

        Args:
            algo_name_to_grouped_agents (Dict[str, List[rl_agent.AbstractAgent]]): the agents to evaluate, grouped by algorithm
            envs (List[rl_environment.Environment]): the environments to evaluate the agents in
            episode_idx (int): the current episode index

        Returns:
            Dict[str, float]: a dictionary of the metrics to return at each evaluation step
        """

        if not episode_idx % self.eval_frequency == 0:
            return {}
        
        metrics_dict = {}
        list_algo_name_and_grouped_agents = list(algo_name_to_grouped_agents.items())
        for algo1_idx, (algo1_name, grouped_agents1) in enumerate(list_algo_name_and_grouped_agents):
            assert len(grouped_agents1) == 2, "This metric only works for 2 player games."
            for (algo2_name, grouped_agents2) in list_algo_name_and_grouped_agents[algo1_idx+1:]:
                faceoff_results_agent1_vs_agent2 = self.evaluate_two_agents(grouped_agents1[0], grouped_agents2[1])
                faceoff_results_agent2_vs_agent1 = self.evaluate_two_agents(grouped_agents2[0], grouped_agents1[1])
                agent1_victory_rate_playing_first = faceoff_results_agent1_vs_agent2["agentA_victory_rate"]
                agent1_draw_rate_playing_first = faceoff_results_agent1_vs_agent2["draw_rate"]
                agent2_victory_rate_playing_first = faceoff_results_agent2_vs_agent1["agentA_victory_rate"]
                agent2_draw_rate_playing_first = faceoff_results_agent2_vs_agent1["draw_rate"]

                # E.g. (PPO and DQN)
                match_name = f"({algo1_name} and {algo2_name})"
                # Metrics where the order of the agents matters
                metrics_dict[f"{match_name}/{algo1_name}_vs_{algo2_name}_victory_rate"] = agent1_victory_rate_playing_first
                metrics_dict[f"{match_name}/{algo1_name}_vs_{algo2_name}_draw"] = agent1_draw_rate_playing_first                   
                metrics_dict[f"{match_name}/{algo2_name}_vs_{algo1_name}_victory_rate"] = agent2_victory_rate_playing_first  # 0.17
                metrics_dict[f"{match_name}/{algo2_name}_vs_{algo1_name}_draw"] = agent2_draw_rate_playing_first
                # Order-averaged metrics
                agent1_victory_rate_playing_second = 1 - agent2_victory_rate_playing_first - agent2_draw_rate_playing_first
                agent1_victory_rate = (agent1_victory_rate_playing_first + agent1_victory_rate_playing_second) / 2
                draw_rate = (agent1_draw_rate_playing_first + agent2_draw_rate_playing_first) / 2
                agent2_victory_rate = 1 - agent1_victory_rate - draw_rate
                metrics_dict[f"{match_name}/{algo1_name}_victory_rate"] = agent1_victory_rate
                metrics_dict[f"{match_name}/draw"] = draw_rate
                metrics_dict[f"{match_name}/{algo2_name}_victory_rate"] = agent2_victory_rate
        
        return metrics_dict
    
    
    def evaluate_two_agents(self, 
        agent_a : rl_agent.AbstractAgent,
        agent_b : rl_agent.AbstractAgent,
        ) -> Dict[str, float]:
        """Evaluate two agents against each other.

        Args:
            agent_a (rl_agent.AbstractAgent): the first agent
            agent_b (rl_agent.AbstractAgent): the second agent
        
        Returns:
            Dict[str, float]: a dictionary of the metrics for the two agents
        """
        env = rl_environment.Environment(self.game_name, **self.game_config)
        evaluated_agents = [agent_a, agent_b]
        
        are_victories = []   # [1(G_0^ep > G_0_adv^ep) | ep in [0, n_episodes_evaluation-1]]
        are_draws = []   # [1(G_0^ep == G_0_adv^) | ep in [0, n_episodes_evaluation-1]]
        
        for episode in range(self.n_episodes_evaluation):
            time_step = env.reset()
            G_0 = 0         # The cumulative reward of agent 1
            G_0_adv = 0     # The cumulative reward of agent 2
            while not time_step.last():
                player_id = time_step.observations["current_player"]
                agent_output = evaluated_agents[player_id].step(time_step, is_evaluation=True)   # Note the evaluation flag. A Q-learner will set epsilon=0 here.
                time_step = env.step([agent_output.action])
                G_0 += time_step.rewards[0]
                G_0_adv += time_step.rewards[1]
            for agent in evaluated_agents:
                agent.step(time_step)
            are_victories.append(int(G_0 > G_0_adv))
            are_draws.append(int(G_0 == G_0_adv))
            
        victory_rate = np.mean(are_victories)
        draw_rate = np.mean(are_draws)
        
        return {
                f"agentA_victory_rate" : victory_rate,
                f"draw_rate" : draw_rate,
            }
    