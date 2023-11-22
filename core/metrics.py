from time import sleep
from typing import List
from matplotlib.pylab import f

from omegaconf import DictConfig
import wandb
from open_spiel.python import rl_agent, rl_environment
import numpy as np



class Metric:
    def __init__(self, config : DictConfig, training_env : rl_environment.Environment) -> None:
        self.training_env = training_env
        self.config = config

    def evaluate(agents):
        raise NotImplementedError



from open_spiel.python.algorithms import random_agent
class VsRandomMetric(Metric):
    """A metric that regularly evaluates the agents against random agents."""
    def __init__(self, 
            config : DictConfig,
            training_env : rl_environment.Environment,
            eval_frequency: int, 
            n_evaluation_episodes: int,
            ) -> None:
        self.eval_frequency = eval_frequency
        self.n_evaluation_episodes = n_evaluation_episodes
        super().__init__(config = config, training_env = training_env)

    def evaluate(self, agents: List[rl_agent.AbstractAgent], episode_idx : int):
        """Evaluate the agents against random agents.
        
        First version : 
        - only focus on the first agent (we could do average, min and max, or all)
        - assume this agent starts to play every game (we could randomize the order)

        Args:
            agents (List[rl_agent.AbstractAgent]): the agents to evaluate
            episode_idx (int): the current episode index
        """
        if episode_idx % self.eval_frequency == 0:
            
            env = self.training_env   # WARNING: should we use a separate environment for evaluation?
            num_players = env.num_players
            num_actions = env.action_spec()["num_actions"]
            eval_agents = [agents[0]] +[random_agent.RandomAgent(player_id, num_actions, "Random Agent") for player_id in range(1, num_players)]

            G_0s = []   # [G_0^ep | ep in [0, n_evaluation_episodes-1]]
            are_victories = []   # [1(G_0^ep > G_0_adv^ep) | ep in [0, n_evaluation_episodes-1]]
            
            for episode in range(self.n_evaluation_episodes):
                time_step = env.reset()
                G_0 = 0
                G_0_adv = 0
                while not time_step.last():
                    player_id = time_step.observations["current_player"]
                    agent_output = eval_agents[player_id].step(time_step, is_evaluation=True)   # Note the evaluation flag. A Q-learner will set epsilon=0 here.
                    time_step = env.step([agent_output.action])
                    G_0 += time_step.rewards[0]
                    G_0_adv += time_step.rewards[1]
                for agent in agents:
                    agent.step(time_step)
                G_0s.append(G_0)
                are_victories.append(int(G_0 > G_0_adv))
                
            mean_G_0 = np.mean(G_0s)
            std_G_0 = np.std(G_0s)
            victory_rate = np.mean(are_victories)
            
            if self.config["do_wandb"]:
                wandb.log({
                    "episode_idx": episode_idx,
                    "mean_reward_vs_random": mean_G_0,
                    "std_reward_vs_random": std_G_0,
                    "victory_percentage_vs_random": victory_rate,
                }, step=episode_idx)
            # if self.config.do_tensorboard_log:
            #     pass # TODO
            # if self.config.do_cli_log:
            #     pass # TODO



from open_spiel.python.bots.human import HumanBot
class VsHumanMetric(Metric):
    
    """A metric that "evaluates" the agents against the human user, or rather allows the human to test the AI"""
    def __init__(self, 
            config : DictConfig,
            training_env : rl_environment.Environment,
            eval_frequency: int, 
            n_evaluation_episodes: int,
            ) -> None:
        assert type(eval_frequency) == int or eval_frequency in ["start", "end"]
        self.eval_frequency = eval_frequency
        self.n_evaluation_episodes = n_evaluation_episodes
        super().__init__(config = config, training_env = training_env)

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

            for episode in range(self.n_evaluation_episodes):
                print(f"VS-Human Evaluation episode {episode+1}/{self.n_evaluation_episodes} :")
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
                            
