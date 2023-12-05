from abc import abstractmethod, ABC
from time import sleep
from omegaconf import DictConfig
from typing import Dict, List, Type
import numpy as np

from open_spiel.python import rl_agent, rl_environment
from open_spiel.python.bots.human import HumanBot

from metrics.base_metric import Metric



class VsHumanMetric(Metric):
    
    """A metric that "evaluates" the agents against the human user, or said otherwise allows the human to test the AI"""
    def __init__(self, 
            config : DictConfig,
            eval_frequency: int, 
            n_episodes_evaluation: int,
            ) -> None:
        assert type(eval_frequency) == int or eval_frequency in ["start", "end"]
        self.eval_frequency = eval_frequency
        self.n_episodes_evaluation = n_episodes_evaluation
        super().__init__(config = config)


    def is_episode_idx_to_evaluate(self, episode_idx):
        if self.eval_frequency == "start":
            return episode_idx == 0
        elif self.eval_frequency == "end":
            return episode_idx == self.config["n_episodes_training"]-1
        else:
            return episode_idx % self.eval_frequency == 0
            
        
    def evaluate(self, agents: List[rl_agent.AbstractAgent], episode_idx : int):
        """Do some game against the human.
        
        First version : 
        - only focus on the first agent (we could fight all agents)
        - assume this agent starts to play

        Args:
            agents (List[rl_agent.AbstractAgent]): the agents to evaluate
            episode_idx (int): the current episode index
        """
        if self.is_episode_idx_to_evaluate(episode_idx=episode_idx):
            env = self.training_env   # WARNING: should we use a separate environment for evaluation?
            num_players = env.num_players
            num_actions = env.action_spec()["num_actions"]
            eval_agents = [agents[player_id] for player_id in range(num_players-1)] + [HumanBot()]

            for episode in range(self.n_episodes_evaluation):
                print(f"VS-Human Evaluation episode {episode+1}/{self.n_episodes_evaluation} :")
                time_step = env.reset()
                while not time_step.last():
                    player_id = time_step.observations["current_player"]
                    print(f"State:\n{env._state}")

                    if player_id == num_players-1:
                        # HumanBot is playing
                        action = eval_agents[player_id].step(env._state)
                        print(f"Human chooses action {action}")
                        time_step = env.step([action])
                    else:
                        # Agent is playing
                        sleep(1)
                        agent_output = eval_agents[player_id].step(time_step, is_evaluation=True)   # Note the evaluation flag. A Q-learner will set epsilon=0 here.
                        print(f"Agent {player_id} chooses action {agent_output.action}")
                        time_step = env.step([agent_output.action])

                print(f"Final state:\n{env._state}")
                print(f"Final rewards: {time_step.rewards}")
                if num_players == 2:
                    if time_step.rewards[0] > time_step.rewards[1]:
                        print("Result : Agent wins ...")
                    elif time_step.rewards[0] < time_step.rewards[1]:
                        print("Result : Human wins !")
                    else:
                        print("Result : Draw.")
        
        return {}   # this metric does not return anything, it only faces the human and prints stuff                 
















from abc import abstractmethod, ABC
from time import sleep
from omegaconf import DictConfig
from typing import Dict, List, Type
import numpy as np

from open_spiel.python import rl_agent, rl_environment
from open_spiel.python.algorithms import random_agent

from metrics.base_metric import Metric, evaluate_each_group_independently



class VsHumanMetric(Metric):
    """A metric that evaluates the agents against human, say otherwise that allows the human to test the AI."""
    def __init__(self, 
            config : DictConfig,
            eval_frequency: int, 
            n_episodes_evaluation: int,
            human_id : int,
            ) -> None:
        self.eval_frequency = eval_frequency
        self.n_episodes_evaluation = n_episodes_evaluation
        self.human_id = human_id
        super().__init__(config = config)


    def is_episode_idx_to_evaluate(self, episode_idx):
        if self.eval_frequency == "start":
            return episode_idx == 0
        elif self.eval_frequency == "end":
            return episode_idx == self.config["n_episodes_training"]-1
        elif self.eval_frequency == "never":
            return False
        elif self.eval_frequency == "always":
            return True
        elif isinstance(self.eval_frequency, int):
            return episode_idx % self.eval_frequency == 0
        else:
            raise NotImplementedError(f"Unrecognized string eval_frequency {self.eval_frequency}")
        
    def evaluate(self, 
            group_names_to_grouped_agents : Dict[str, List[rl_agent.AbstractAgent]],
            group_names_to_envs : Dict[str, rl_environment.Environment],
            episode_idx : int,
            ) -> Dict[str, float]:
        """Evaluate each agents against human.

        Args:
            group_names_to_grouped_agents (Dict[str, List[rl_agent.AbstractAgent]]): the agents to evaluate, grouped by algorithm
            group_names_to_envs (Dict[str, rl_environment.Environment]): the environments to evaluate the agents in
            episode_idx (int): the current episode index
            
        Returns:
            Dict[str, float]: a dictionary of the metrics to return at each evaluation step, with
        """
        if not self.is_episode_idx_to_evaluate(episode_idx=episode_idx):
            return {}        

        human_bot = HumanBot()
        for episode in range(self.n_episodes_evaluation):
            print(f"VS-Human Evaluation episode {episode+1}/{self.n_episodes_evaluation} :")
            
            # Human pick the opponent
            # If there is only one algorithm, we don't ask the human to choose
            if len(group_names_to_grouped_agents) == 1:
                group_name = list(group_names_to_grouped_agents.keys())[0]
                break
            # Else, we ask the human to choose
            else:
                while True:
                    group_name = input(f"Choose the opponent algorithm among {list(group_names_to_grouped_agents.keys())} : ")
                    if group_name in group_names_to_grouped_agents:
                        break
                    else:
                        print("Invalid opponent algorithm name.")
            print(f"Opponent algorithm : {group_name}")
            
            env = group_names_to_envs[group_name]
            evaluated_agents = group_names_to_grouped_agents[group_name]
            num_players = env.num_players
            
            time_step = env.reset()
            while not time_step.last():
                player_id = time_step.observations["current_player"]
                print(f"State:\n{env._state}")

                if player_id == self.human_id:
                    # HumanBot is playing
                    action = human_bot.step(env._state)
                    print(f"Human chooses action {action}")
                    time_step = env.step([action])
                else:
                    # Agent is playing
                    sleep(1)
                    agent_output = evaluated_agents[player_id].step(time_step, is_evaluation=True)   # Note the evaluation flag. A Q-learner will set epsilon=0 here.
                    print(f"Agent {player_id} chooses action {agent_output.action}")
                    time_step = env.step([agent_output.action])

            print(f"Final state:\n{env._state}")
            print(f"Final rewards: {time_step.rewards}")
            if num_players == 2:
                if time_step.rewards[0] > time_step.rewards[1]:
                    print("Result : Agent wins ...")
                elif time_step.rewards[0] < time_step.rewards[1]:
                    print("Result : Human wins !")
                else:
                    print("Result : Draw.")
    
        return {}   # this metric does not return anything, it only faces the human and prints stuff    

        
        
        