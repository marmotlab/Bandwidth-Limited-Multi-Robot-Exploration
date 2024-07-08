import matplotlib.pyplot as plt
import torch
from env import Env
from agent import Agent
from utils import *
from ground_truth_node_manager import Ground_truth_node_manager
import matplotlib.gridspec as gridspec
from test_parameter import *

if not os.path.exists(gifs_path):
    os.makedirs(gifs_path)

class TestWorker:
    def __init__(self, meta_agent_id, policy_net, global_step, device='cpu', save_image=False, greedy=False):
        self.meta_agent_id = meta_agent_id
        self.global_step = global_step
        self.save_image = save_image
        self.device = device
        self.greedy = greedy

        self.env = Env(global_step, plot=self.save_image, test=True)
        self.n_agent = TEST_N_AGENTS
        self.ground_truth_node_manager = None
        self.robot_list = [Agent(i, policy_net, deepcopy(self.env), deepcopy(self.ground_truth_node_manager), self.device, self.save_image) for i in
                           range(TEST_N_AGENTS)]

        self.perf_metrics = dict()

    def run_episode(self):
        done = False
        for robot in self.robot_list:
            robot.update_graph(self.env.belief_info, deepcopy(self.env.robot_locations[robot.id]))
        for robot in self.robot_list:
            robot.update_planning_state(self.env.robot_locations)

        for i in range(MAX_EPISODE_STEP):
            selected_locations = []
            dist_list = []
            local_observations = []

            for robot in self.robot_list:
                local_observations.append(robot.get_no_padding_observation())
                local_observation = local_observations[robot.id]

                local_node_inputs, local_node_padding_mask, local_edge_mask, current_local_index, current_local_edge, local_edge_padding_mask = local_observation
                current_coord = torch.tensor(robot.location, dtype=torch.float32, device=self.device).reshape(1, 1, 2)
                enhanced_node_feature, current_state_feature = robot.policy_net.get_current_state_feature(local_node_inputs, local_node_padding_mask, local_edge_mask, current_local_index,current_coord)
                # send detached current_state_feature to other robots
                self.send_msg(current_state_feature.detach(), robot.id)


            for robot in self.robot_list:
                local_observation = local_observations[robot.id]
                local_node_inputs, local_node_padding_mask, local_edge_mask, current_local_index, current_local_edge, local_edge_padding_mask = local_observation

                stacked_msg = robot.get_stacked_msg()

                next_location, _, _ = robot.select_next_waypoint(local_observation, stacked_msg)
                
                selected_locations.append(next_location)
                dist_list.append(np.linalg.norm(next_location - robot.location))

            selected_locations = np.array(selected_locations).reshape(-1, 2)
            arriving_sequence = np.argsort(np.array(dist_list))
            selected_locations_in_arriving_sequence = np.array(selected_locations)[arriving_sequence]

            for j, selected_location in enumerate(selected_locations_in_arriving_sequence):
                solved_locations = selected_locations_in_arriving_sequence[:j]
                while selected_location[0] + selected_location[1] * 1j in solved_locations[:,
                                                                          0] + solved_locations[:, 1] * 1j:
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

            for robot, next_location in zip(self.robot_list, selected_locations):
                self.env.step(next_location, robot.id)
                robot.env.step(next_location, robot.id)
                robot.update_graph(robot.env.belief_info, deepcopy(robot.env.robot_locations[robot.id]))

            for robot in self.robot_list:
                robot.update_planning_state(self.env.robot_locations)

            if self.env.explored_rate >= 0.95:
                done = True

            if self.save_image:
                self.plot_local_env(i)

            if done:
                break

        # save metrics
        self.perf_metrics['travel_dist'] = max([robot.travel_dist for robot in self.robot_list])
        self.perf_metrics['success_rate'] = done


        # save gif
        if self.save_image:
            make_gif(gifs_path, self.global_step, self.env.frame_files, self.env.explored_rate)

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
                    # for plotting, if any element in utility is -1, set it to 0
                    untility[untility == -1] = 0
                    ax.scatter(nodes[:, 0], nodes[:, 1], c=untility, s=5, zorder=2)
                robot_cell = get_cell_position_from_coords(robot.location, robot.global_map_info)
                ax.plot(robot_cell[0], robot_cell[1], c+'o', markersize=4, zorder=5)
            
            ax.axis('off')
            ax.set_title('Robot {}'.format(i))
            
            
        plt.tight_layout()
        # plt.show()
        plt.savefig('{}/{}_{}_samples.png'.format(gifs_path, self.global_step, step), dpi=150)
        frame = '{}/{}_{}_samples.png'.format(gifs_path, self.global_step, step)
        self.env.frame_files.append(frame)
        plt.close()