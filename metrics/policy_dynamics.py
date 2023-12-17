from abc import abstractmethod, ABC
from collections import defaultdict
import os
from time import sleep
from omegaconf import DictConfig
from typing import Dict, List, Type, Union
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from open_spiel.python import rl_agent, rl_environment
from open_spiel.python.algorithms import random_agent

from metrics.base_metric import Metric, evaluate_each_group_independently
from open_spiel.python.rl_agent_policy import JointRLAgentPolicy


def plot_trajectory(trajectory, color, label, marker, size):
    x = trajectory[:, 0]
    y = trajectory[:, 1]
    plt.plot(x, y, color=color, label=label, markersize=size)
    plt.scatter(x[-1], y[-1], color=color, marker=marker, s=size)

# Function to convert probability values to Cartesian coordinates
def convert_to_cartesian(p1, p2, p3):
    x = p1 * 0 + p2 * 1 + p3 * 0.5
    y = p1 * 0 + p2 * 0 + p3 * np.sqrt(3) / 2
    return x, y
    
    

class PolicyDynamicsMetric(Metric):
    """Adapted for one-shot games. Log the action probabilities and keep track of the trajectory of the policies to create a figure of the policy dynamics.
    Currently only works for 2-player one-shot games with 3 actions (rock-paper-scissors).
    """
    def __init__(self, 
            config : DictConfig,
            eval_frequency : int = 1000,
            ) -> None:
        self.eval_frequency = eval_frequency
        self.group_names_to_agent_id_to_distribution_trajectory : Dict[str, Dict[int, List[np.ndarray]]] = defaultdict(lambda : defaultdict(list))
        # Example self.group_names_to_agent_id_to_policies_trajectories["(dqn, dq,)"][0] = [np.array([0.4, 0.3, 0.3]), np.array([0.5, 0.2, 0.3])
        super().__init__(config = config)

    def evaluate(self, 
            group_names_to_grouped_agents : Dict[str, List[rl_agent.AbstractAgent]],
            group_names_to_envs : List[rl_environment.Environment],
            episode_idx : int,
            ) -> Dict[str, float]:
        
        metrics_dict = {}
        
        # Add the current action probabilities to the trajectory
        env : rl_environment.Environment = list(group_names_to_envs.values())[0]
        for group_name, grouped_agents in group_names_to_grouped_agents.items():
            env.reset()
            agents_dict = {agent_id : agent for agent_id, agent in enumerate(grouped_agents)}
            policy = JointRLAgentPolicy(
                game = env.game,
                agents = agents_dict,
                use_observation=env.use_observation,
            )
            # Get the action probabilities of the policy of the first player
            action_distribution_0_dict = policy.action_probabilities(env.get_state, player_id=0)
            metrics_dict.update({f"Policy Dynamics/{group_name} agent 0 probability of action {action}" : probability for action, probability in action_distribution_0_dict.items()})
            action_distribution_0_list = np.array([action_distribution_0_dict[0], action_distribution_0_dict[1], action_distribution_0_dict[2]])
            self.group_names_to_agent_id_to_distribution_trajectory[group_name][0].append(action_distribution_0_list)
            # Play a random action just to get the next state
            time_step = env.step([0])
            # Get the action probabilities of the policy of the first player in the next state
            action_distribution_1_dict = policy.action_probabilities(env.get_state, player_id=1)
            metrics_dict.update({f"Policy Dynamics/{group_name} agent 1 probability of action {action}" : probability for action, probability in action_distribution_1_dict.items()})
            action_distribution_1_list = np.array([action_distribution_1_dict[0], action_distribution_1_dict[1], action_distribution_1_dict[2]])
            self.group_names_to_agent_id_to_distribution_trajectory[group_name][1].append(action_distribution_1_list)
        
        # If it is time to evaluate, plot the trajectories
        if episode_idx % self.eval_frequency == 0:
            
            print(f"[PolicyDynamicsMetric] Plotting policy dynamics at episode {episode_idx}")
            # Do one plot/figure for each group
            for group_name, agent_id_to_distribution_trajectory in self.group_names_to_agent_id_to_distribution_trajectory.items():
                
                # Plotting
                fig, ax = plt.subplots()
                ax.set_aspect('equal', 'box')
                ax.set_xlim(0, 1)
                ax.set_ylim(0, np.sqrt(3) / 2)
                ax.axis('off')
                # Create triangle
                triangle = Polygon([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2]], closed=True, edgecolor='black', facecolor='none')
                ax.add_patch(triangle)
                # Add labels to each corner
                corner_labels = ['Rock (1)', 'Paper (2)', 'Scissors (3)']
                for i, label in enumerate(corner_labels):
                    eps = 0.05
                    ax.text([0-eps, 1+eps, 0.5][i], [0-eps, 0-eps, eps + np.sqrt(3) / 2][i], label, horizontalalignment='center', verticalalignment='center')
                # Add title
                ax.set_title(f'Policy Dynamics of {group_name}', loc='left')
                # Add Nash Equilibrium at the center as a text and a big green point
                ax.scatter(0.5, np.sqrt(3) / 6, color='green', s=200)
                ax.text(0.5 + eps, eps + np.sqrt(3) / 6, 'NE', horizontalalignment='center', verticalalignment='center', color='green')
                
                agent_id_to_colors = {0 : 'red', 1 : 'blue'}
                agent_id_to_shapes = {0 : 'o', 1 : 's'}
                agent_id_to_sizes = {0 : 20, 1 : 5}
                for agent_id, distribution_trajectory in agent_id_to_distribution_trajectory.items():
                    # Convert probabilities to Cartesian coordinates
                    trajectory_cartesian = np.array([convert_to_cartesian(p1, p2, p3) for p1, p2, p3 in distribution_trajectory])

                    # Plot the trajectory
                    plot_trajectory(trajectory_cartesian, color=agent_id_to_colors[agent_id], label=f'Agent {agent_id}', marker=agent_id_to_shapes[agent_id], size=agent_id_to_sizes[agent_id])

                    # ax.text(trajectory_cartesian[0, 0], trajectory_cartesian[0, 1], f'ep=0 (id={agent_id})', ha='right', va='bottom', color='black', weight='bold')
                    # ax.text(trajectory_cartesian[-1, 0], trajectory_cartesian[-1, 1], f'ep={episode_idx} (id={agent_id})', ha='left', va='top', color='black', weight='bold')

                # Save plot
                plt.legend()
                for fig_name in [
                    f"logs/{self.config['game_theory_algorithm'].run_name}/policy_dynamics_{group_name}/episode_{episode_idx}.png",
                    f"logs/{self.config['game_theory_algorithm'].run_name}/policy_dynamics_{group_name}_last.png",
                    f"logs/policy_dynamics_last.png",
                ]:
                    os.makedirs(os.path.dirname(fig_name), exist_ok=True)
                    plt.savefig(fig_name)

        return metrics_dict