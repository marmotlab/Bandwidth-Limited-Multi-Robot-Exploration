import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from copy import deepcopy

from env import Env
from agent import Agent
from parameter import *
from utils import *
from ground_truth_node_manager import Ground_truth_node_manager
import pickle

if not os.path.exists(gifs_path):
    os.makedirs(gifs_path)


class Multi_agent_worker:
    def __init__(self, meta_agent_id, policy_net, global_step, device='cpu', save_image=False):
        self.meta_agent_id = meta_agent_id
        self.global_step = global_step
        self.save_image = save_image
        self.device = device

        self.env = Env(global_step, plot=self.save_image)
        map_path = self.env.map_path

        map_name = map_path.split('/')[-1].split('.')[0]
        folder_name = map_path.split('/')[-2] + '_data'
        data_path = f'{folder_name}/{map_name}'
        
        self.n_agent = N_AGENTS
        try:
            with open(f'{data_path}.pkl', 'rb') as f:
                self.ground_truth_node_manager = pickle.load(f)
            print("Ground truth node manager loaded")
        except FileNotFoundError:
            print("File not found, creating new ground truth node manager")
            self.ground_truth_node_manager = Ground_truth_node_manager(self.env.ground_truth_info, plot=True)
            # save ground truth node manager
            with open(f'{data_path}.pkl', 'wb') as f:
                pickle.dump(self.ground_truth_node_manager, f, pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print("Error: ", e)
            self.ground_truth_node_manager = Ground_truth_node_manager(self.env.ground_truth_info, plot=True)
            with open(f'{data_path}.pkl', 'wb') as f:
                pickle.dump(self.ground_truth_node_manager, f, pickle.HIGHEST_PROTOCOL)

        self.robot_list = [Agent(i, policy_net, deepcopy(self.env), deepcopy(self.ground_truth_node_manager), self.device, self.save_image) for i in
                           range(N_AGENTS)]
        
        
        self.episode_buffer = []
        self.perf_metrics = dict()
        for _ in range(19):
            self.episode_buffer.append([])
            
        self.ground_truth_episode_buffer = []
        for _ in range(12):
            self.ground_truth_episode_buffer.append([])
 
    def run_episode(self):
        done = False

        self.update_ground_truth_graph(self.env.belief_info)
        for robot in self.robot_list:
            robot.update_graph(robot.env.belief_info, deepcopy(robot.env.robot_locations[robot.id]))
            robot.update_ground_truth_graph(deepcopy(self.ground_truth_node_manager))

        # update all robots' planning state
        for robot in self.robot_list:    
            robot.update_planning_state(robot.env.robot_locations)
            robot.update_ground_truth_planning_state(robot.env.robot_locations)

        for i in range(MAX_EPISODE_STEP):
            selected_locations = []
            dist_list = []
            next_node_index_list = []
            local_observations = []
            next_local_observations = []
            
            # for qnet
            ground_truth_observations = [] 
            next_ground_truth_observations = []

            # send msg
            for robot in self.robot_list:
                local_observations.append(robot.get_local_observation())
                local_observation = local_observations[robot.id]

                local_node_inputs, local_node_padding_mask, local_edge_mask, current_local_index, current_local_edge, local_edge_padding_mask = local_observation
                current_coord = torch.tensor(robot.location, dtype=torch.float32, device=self.device).reshape(1, 1, 2)
                enhanced_node_feature, current_state_feature = robot.policy_net.get_current_state_feature(local_node_inputs, local_node_padding_mask, local_edge_mask, current_local_index,current_coord)
                # send detached current_state_feature to other robots
                self.send_msg(current_state_feature.detach(), robot.id)

            for robot in self.robot_list:
                # ground truth observation
                ground_truth_obsesrvation = robot.get_ground_truth_observation()
                ground_truth_observations.append(ground_truth_obsesrvation)
                
                robot.save_ground_truth_observation(ground_truth_obsesrvation)
                
                local_observation = local_observations[robot.id]
                local_node_inputs, local_node_padding_mask, local_edge_mask, current_local_index, current_local_edge, local_edge_padding_mask = local_observation

                stacked_msg = robot.get_stacked_msg()

                robot.save_observation(local_observation, stacked_msg)

                next_location, next_node_index, action_index = robot.select_next_waypoint(local_observation, stacked_msg)
                robot.save_action(action_index)
                
                node = robot.local_node_manager.local_nodes_dict.find((robot.location[0], robot.location[1]))
                check = np.array(node.data.neighbor_list)
                assert next_location[0] + next_location[1] * 1j in check[:, 0] + check[:, 1] * 1j, print(next_location,
                                                                                                         robot.location,
                                                                                                         node.data.neighbor_list)
                assert next_location[0] != robot.location[0] or next_location[1] != robot.location[1]

                selected_locations.append(next_location)
                dist_list.append(np.linalg.norm(next_location - robot.location))
                next_node_index_list.append(next_node_index)
            

            selected_locations = np.array(selected_locations).reshape(-1, 2)
            arriving_sequence = np.argsort(np.array(dist_list))
            selected_locations_in_arriving_sequence = np.array(selected_locations)[arriving_sequence]

            for j, selected_location in enumerate(selected_locations_in_arriving_sequence):
                solved_locations = selected_locations_in_arriving_sequence[:j]
                while selected_location[0] + selected_location[1] * 1j in solved_locations[:, 0] + solved_locations[:, 1] * 1j:
                    id = arriving_sequence[j]
                    nearby_nodes = self.robot_list[id].local_node_manager.local_nodes_dict.nearest_neighbors(
                        selected_location.tolist(), 25)
                    for node in nearby_nodes:
                        coords = node.data.coords
                        if coords[0] + coords[1] * 1j in solved_locations[:, 0] + solved_locations[:, 1] * 1j:
                            continue
                        selected_location = coords
                        break

                    selected_locations_in_arriving_sequence[j] = selected_location
                    selected_locations[id] = selected_location
            
            reward_list = []
            for robot, next_location in zip(self.robot_list, selected_locations):   
                
                # node reward
                ground_truth_node_coords_to_check = robot.ground_truth_node_coords[:, 0] + robot.ground_truth_node_coords[:, 1] * 1j
                node_index = np.argwhere(ground_truth_node_coords_to_check == next_location[0] + next_location[1] * 1j)[0]
                individual_reward = robot.ground_truth_node_utility[node_index]

                # momentum reward
                direction = next_location - robot.location
                # normalize
                dist = np.linalg.norm(direction)
                if dist == 0:
                    momentum = np.zeros(2)
                else:
                    momentum = direction / dist
                
                momentum_reward = np.dot(momentum, robot.momentum) * 0.1

                reward_list.append(individual_reward + momentum_reward)
                
                self.env.step(next_location, robot.id)

                # udpate robot's map
                robot.env.step(next_location, robot.id)
                robot.momentum = momentum
                robot.update_graph(robot.env.belief_info, deepcopy(robot.env.robot_locations[robot.id]))
            
            self.update_ground_truth_graph(self.env.belief_info)
            for robot in self.robot_list:
                robot.update_ground_truth_graph(deepcopy(self.ground_truth_node_manager))

            # for robot's env to update the other robots' location
            for robot in self.robot_list:
                for other_robot in self.robot_list:
                    if other_robot.id == robot.id:
                        continue
                    robot.env.update_robot_location(other_robot.location, other_robot.id)

            if self.env.explored_rate >= 0.9:
                done = True

            team_reward = self.env.calculate_reward() - 0.5
            if done:
                team_reward += 10
            
            for agent_id in range(len(self.robot_list)):
                self.robot_list[agent_id].save_reward(reward_list[agent_id][0] + team_reward)
                self.robot_list[agent_id].update_planning_state(self.robot_list[agent_id].env.robot_locations)
                self.robot_list[agent_id].update_ground_truth_planning_state(self.robot_list[agent_id].env.robot_locations)
                self.robot_list[agent_id].save_done(done)
                

            if self.global_step % 500 > 480 or self.global_step % 500 == 0:
                self.save_image = True
            else:
                self.save_image = False

            if self.save_image:
                self.plot_local_env(i)

            if done:
                if self.save_image:
                    self.plot_local_env(i + 1)
                break

        # save metrics
        self.perf_metrics['travel_dist'] = max([robot.travel_dist for robot in self.robot_list])
        self.perf_metrics['explored_rate'] = self.env.explored_rate
        self.perf_metrics['success_rate'] = done

        # save episode buffer
        temp_next_msgs = []
        for robot in self.robot_list:
            next_local_observations.append(robot.get_local_observation())
            next_local_observation = next_local_observations[robot.id]
            next_local_node_inputs, next_local_node_padding_mask, next_local_edge_mask, next_current_local_index, next_current_local_edge, next_local_edge_padding_mask = next_local_observation

            current_coord = torch.tensor(robot.location, dtype=torch.float32, device=self.device).reshape(1, 1, 2)
            enhanced_node_feature, next_current_state_feature = robot.policy_net.get_current_state_feature(next_local_node_inputs, next_local_node_padding_mask, next_local_edge_mask, next_current_local_index,current_coord)
            temp_next_msgs.append(next_current_state_feature.detach())

        for robot in self.robot_list:
            next_local_observation = next_local_observations[robot.id]
            next_ground_truth_observation = robot.get_ground_truth_observation() 
            next_ground_truth_observations.append(next_ground_truth_observation)
            
            next_stacked_msg = self.stack_msgs(temp_next_msgs, robot.id)
            robot.save_next_observations(next_local_observation, next_stacked_msg)
            robot.save_ground_truth_observations(next_ground_truth_observation)
            for i in range(len(self.episode_buffer)):
                self.episode_buffer[i] += robot.episode_buffer[i]
            for i in range(len(self.ground_truth_episode_buffer)):
                self.ground_truth_episode_buffer[i] += robot.ground_truth_episode_buffer[i]
        
        # save gif
        if self.save_image:
            make_gif(gifs_path, self.global_step, self.env.frame_files, self.env.explored_rate)


    def plot_local_env(self, step):
        plt.switch_backend('agg')
        plt.style.use('fast')
        fig = plt.figure(figsize=(15, 5))
        gs = gridspec.GridSpec(2, 4)
        ax1 = fig.add_subplot(gs[0:2, 0:2])
        ax1.imshow(self.env.robot_belief, cmap='gray')
        ax1.axis('off')
        color_list = ['r', 'b', 'g', 'y']
        frontiers = get_frontier_in_map(self.env.belief_info)
        frontiers = get_cell_position_from_coords(frontiers, self.env.belief_info).reshape(-1, 2)
        ax1.scatter(frontiers[:, 0], frontiers[:, 1], c='r', s=1)
        for robot in self.robot_list:
            c = color_list[robot.id]
            robot_cell = get_cell_position_from_coords(robot.location, robot.global_map_info)
            ax1.plot(robot_cell[0], robot_cell[1], c+'o', markersize=16, zorder=5)
            ax1.plot((np.array(robot.trajectory_x) - robot.global_map_info.map_origin_x) / robot.cell_size,
                     (np.array(robot.trajectory_y) - robot.global_map_info.map_origin_y) / robot.cell_size, c,
                     linewidth=2, zorder=1)
        
        ax1.set_title('Explored ratio: {:.4g}  Travel distance: {:.4g}'.format(self.env.explored_rate,
                                                                              max([robot.travel_dist for robot in
                                                                                   self.robot_list])))
            
        for i in range(self.n_agent):
            ax = fig.add_subplot(gs[i // 2, 2 + i % 2])
            ax.imshow(self.robot_list[i].local_map_info.map, cmap='gray')
            frontiers = get_frontier_in_map(self.robot_list[i].env.belief_info)
            frontiers = get_cell_position_from_coords(frontiers, self.robot_list[i].env.belief_info).reshape(-1, 2)
            ax.scatter(frontiers[:, 0], frontiers[:, 1], c='r', s=1)
            for robot in self.robot_list:
                c = color_list[robot.id]
                if robot.id == i:
                    nodes = get_cell_position_from_coords(robot.local_node_coords, robot.global_map_info)
                    ax.imshow(robot.global_map_info.map, cmap='gray')
                    ax.axis('off')
                    untility = robot.utility
                    untility[untility == -1] = 0
                    ax.scatter(nodes[:, 0], nodes[:, 1], c=untility, s=5, zorder=2)
                robot_cell = get_cell_position_from_coords(robot.location, robot.global_map_info)
                ax.plot(robot_cell[0], robot_cell[1], c+'o', markersize=4, zorder=5)
            
            ax.axis('off')
            ax.set_title('Robot {}'.format(i))
            
            
        plt.tight_layout()
        plt.savefig('{}/{}_{}_samples.png'.format(gifs_path, self.global_step, step), dpi=150)
        frame = '{}/{}_{}_samples.png'.format(gifs_path, self.global_step, step)
        self.env.frame_files.append(frame)
        plt.close()
    
    def send_msg(self, msg, robot_id):
        for robot in self.robot_list:
            if len(robot.msgs[robot_id]) > 5:
                # delete the oldest msg
                robot.msgs[robot_id].pop(0)
            robot.msgs[robot_id].append(msg.clone())

    def stack_msgs(self, msgs, robot_id):
        stacked_msg = []
        for i in range(self.n_agent):
            if i == robot_id:
                continue
            stacked_msg.append(msgs[i])
        stacked_msg = torch.stack(stacked_msg, dim=1).squeeze(2)
        return stacked_msg.clone()
    
    def update_ground_truth_graph(self, map_info):
        self.ground_truth_node_manager.update_ground_truth_graph(map_info)
