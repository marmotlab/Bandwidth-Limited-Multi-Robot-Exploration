import copy
import numpy as np
import torch
from utils import *
from parameter import *
from test_parameter import *
from local_node_manager_quadtree import Local_node_manager


class Agent:
    def __init__(self, id, policy_net, env, ground_truth_node_manager, device='cpu', plot=False):
        if env.test:
            N_AGENTS = TEST_N_AGENTS
        self.id = id
        self.device = device
        self.plot = plot
        self.policy_net = policy_net

        # location and global map
        self.location = None
        self.global_map_info = None
        self.local_center = None

        # local map related parameters
        self.cell_size = CELL_SIZE
        self.downsample_size = NODE_RESOLUTION  # cell
        self.downsampled_cell_size = self.cell_size * self.downsample_size  # meter
        self.local_map_size = LOCAL_MAP_SIZE  # meter
        self.extended_local_map_size = EXTENDED_LOCAL_MAP_SIZE

        # local map and extended local map
        self.local_map_info = None
        self.extended_local_map_info = None

        # local frontiers
        self.local_frontier = None

        # local  node managers
        self.local_node_manager = Local_node_manager(plot=self.plot)
        
        # env
        self.env = env
        
        # ground truth node manager
        self.ground_truth_node_manager = ground_truth_node_manager
        self.ground_truth_node_coords = None
        self.ground_truth_node_utility = None
        self.ground_truth_node_guidepost = None
        self.ground_truth_node_occupancy = None
        self.ground_truth_current_index = None
        self.ground_truth_adjacent_matrix = None
        self.ground_truth_neighbor_indices = None
        
        # local graph
        self.local_node_coords, self.utility, self.guidepost, self.occupancy = None, None, None, None
        self.current_local_index, self.local_adjacent_matrix, self.local_neighbor_indices = None, None, None

        # msg
        self.msgs =[[] for _ in range(N_AGENTS)]

        # momentum
        self.momentum = np.zeros(2) 

        self.travel_dist = 0

        self.episode_buffer = []
        for _ in range(19):
            self.episode_buffer.append([])
            
        self.ground_truth_episode_buffer = []
        for _ in range(12):
            self.ground_truth_episode_buffer.append([])

        if self.plot:
            self.trajectory_x = []
            self.trajectory_y = []


    def update_global_map(self, global_map_info):
        # no need in training because of shallow copy
        self.global_map_info = global_map_info

    def update_local_map(self):
        self.local_map_info = self.get_local_map(self.location)
        self.extended_local_map_info = self.get_extended_local_map(self.location)

    def update_location(self, location):
        if self.location is None:
            self.location = location

        dist = np.linalg.norm(self.location - location)
        self.travel_dist += dist

        self.location = location
        node = self.local_node_manager.local_nodes_dict.find((location[0], location[1]))
        if node:
            node.data.set_visited()
        if self.plot:
            self.trajectory_x.append(location[0])
            self.trajectory_y.append(location[1])

    def update_local_frontiers(self):
        self.local_frontier = get_frontier_in_map(self.extended_local_map_info)

    def update_graph(self, global_map_info, location):
        self.update_global_map(global_map_info)
        self.update_location(location)
        self.update_local_map()
        self.update_local_frontiers()
        self.local_node_manager.update_local_graph(self.location,
                                                   self.local_frontier,
                                                   self.local_map_info,
                                                   self.extended_local_map_info)
        
    def update_ground_truth_graph(self, ground_truth_node_manager):
        self.ground_truth_node_manager = ground_truth_node_manager

    def update_planning_state(self, robot_locations):
        self.local_node_coords, self.utility, self.guidepost, self.occupancy, self.local_adjacent_matrix, self.current_local_index, self.local_neighbor_indices = \
            self.local_node_manager.get_all_node_graph(self.location, robot_locations)
        
    def update_ground_truth_planning_state(self, robot_locations):
        self.ground_truth_node_coords, self.ground_truth_node_utility, self.ground_truth_node_guidepost, self.ground_truth_node_occupancy, self.ground_truth_adjacent_matrix, self.ground_truth_current_index, self.ground_truth_neighbor_indices = \
            self.ground_truth_node_manager.get_all_node_graph(self.location, robot_locations)
    
    def get_local_observation(self):
        local_node_coords = self.local_node_coords
        local_node_utility = self.utility.reshape(-1, 1)
        local_node_guidepost = self.guidepost.reshape(-1, 1)
        local_node_occupancy = self.occupancy.reshape(-1, 1)
        current_local_index = self.current_local_index
        local_edge_mask = self.local_adjacent_matrix
        current_local_edge = self.local_neighbor_indices
        n_local_node = local_node_coords.shape[0]

        current_local_node_coords = local_node_coords[self.current_local_index]
        local_node_coords = np.concatenate((local_node_coords[:, 0].reshape(-1, 1) - current_local_node_coords[0],
                                            local_node_coords[:, 1].reshape(-1, 1) - current_local_node_coords[1]),
                                           axis=-1) / LOCAL_MAP_SIZE
        local_node_utility = local_node_utility
        local_node_inputs = np.concatenate((local_node_coords, local_node_utility, local_node_guidepost,local_node_occupancy), axis=1)
        local_node_inputs = torch.FloatTensor(local_node_inputs).unsqueeze(0).to(self.device)

        if local_node_coords.shape[0] >= LOCAL_NODE_PADDING_SIZE:
            print("out of padding")
            print("local_node_coords.shape: ", local_node_coords.shape[0])
        assert local_node_coords.shape[0] < LOCAL_NODE_PADDING_SIZE
        padding = torch.nn.ZeroPad2d((0, 0, 0, LOCAL_NODE_PADDING_SIZE - n_local_node))
        local_node_inputs = padding(local_node_inputs)

        local_node_padding_mask = torch.zeros((1, 1, n_local_node), dtype=torch.int16).to(self.device)
        local_node_padding = torch.ones((1, 1, LOCAL_NODE_PADDING_SIZE - n_local_node), dtype=torch.int16).to(
            self.device)
        local_node_padding_mask = torch.cat((local_node_padding_mask, local_node_padding), dim=-1)

        current_local_index = torch.tensor([current_local_index]).reshape(1, 1, 1).to(self.device)

        local_edge_mask = torch.tensor(local_edge_mask).unsqueeze(0).to(self.device)

        padding = torch.nn.ConstantPad2d(
            (0, LOCAL_NODE_PADDING_SIZE - n_local_node, 0, LOCAL_NODE_PADDING_SIZE - n_local_node), 1)
        local_edge_mask = padding(local_edge_mask)

        current_in_edge = np.argwhere(current_local_edge == self.current_local_index)[0][0]
        current_local_edge = torch.tensor(current_local_edge).unsqueeze(0)
        k_size = current_local_edge.size()[-1]
        padding = torch.nn.ConstantPad1d((0, LOCAL_K_SIZE - k_size), 0)
        current_local_edge = padding(current_local_edge)
        current_local_edge = current_local_edge.unsqueeze(-1)

        local_edge_padding_mask = torch.zeros((1, 1, k_size), dtype=torch.int16).to(self.device)
        local_edge_padding_mask[0, 0, current_in_edge] = 1
        padding = torch.nn.ConstantPad1d((0, LOCAL_K_SIZE - k_size), 1)
        local_edge_padding_mask = padding(local_edge_padding_mask)

        return [local_node_inputs, local_node_padding_mask, local_edge_mask, current_local_index, current_local_edge, local_edge_padding_mask]

    def get_ground_truth_observation(self):
        local_node_coords = self.ground_truth_node_coords
        local_node_utility = self.ground_truth_node_utility.reshape(-1, 1)
        local_node_guidepost = self.ground_truth_node_guidepost.reshape(-1, 1)
        local_node_occupancy = self.ground_truth_node_occupancy.reshape(-1, 1)
        current_local_index = self.ground_truth_current_index
        local_edge_mask = self.ground_truth_adjacent_matrix
        current_local_edge = self.ground_truth_neighbor_indices
        n_local_node = local_node_coords.shape[0]
        
        current_local_node_coords = local_node_coords[self.ground_truth_current_index]
        
        local_node_coords = np.concatenate((local_node_coords[:, 0].reshape(-1, 1) - current_local_node_coords[0],
                                            local_node_coords[:, 1].reshape(-1, 1) - current_local_node_coords[1]),
                                           axis=-1) / LOCAL_MAP_SIZE
        local_node_utility = local_node_utility
        local_node_inputs = np.concatenate((local_node_coords, local_node_utility, local_node_guidepost,local_node_occupancy), axis=1)
        local_node_inputs = torch.FloatTensor(local_node_inputs).unsqueeze(0).to(self.device)
        
        if local_node_coords.shape[0] >= LOCAL_NODE_PADDING_SIZE:
            print("out of padding")
            print("local_node_coords.shape: ", local_node_coords.shape[0])
        assert local_node_coords.shape[0] < LOCAL_NODE_PADDING_SIZE
        padding = torch.nn.ZeroPad2d((0, 0, 0, LOCAL_NODE_PADDING_SIZE - n_local_node))
        local_node_inputs = padding(local_node_inputs)
        
        local_node_padding_mask = torch.zeros((1, 1, n_local_node), dtype=torch.int16).to(self.device)
        local_node_padding = torch.ones((1, 1, LOCAL_NODE_PADDING_SIZE - n_local_node), dtype=torch.int16).to(
            self.device)
        
        local_node_padding_mask = torch.cat((local_node_padding_mask, local_node_padding), dim=-1)
        current_local_index = torch.tensor([current_local_index]).reshape(1, 1, 1).to(self.device)
        
        local_edge_mask = torch.tensor(local_edge_mask).unsqueeze(0).to(self.device)
        padding = torch.nn.ConstantPad2d(
            (0, LOCAL_NODE_PADDING_SIZE - n_local_node, 0, LOCAL_NODE_PADDING_SIZE - n_local_node), 1)
        local_edge_mask = padding(local_edge_mask)

        neighbor_locations = self.local_node_coords[self.local_neighbor_indices]
        neighbor_locations = self.local_node_coords[self.local_neighbor_indices][:, 0] + self.local_node_coords[self.local_neighbor_indices][:, 1] * 1j
        ground_truth_node_coords_to_check = self.ground_truth_node_coords[:, 0] + self.ground_truth_node_coords[:, 1] * 1j
        neighbor_indices_in_ground_truth_node_manager = []
        for neighbor_location in neighbor_locations:
            neighbor_index = np.argwhere(ground_truth_node_coords_to_check == neighbor_location)
            neighbor_indices_in_ground_truth_node_manager.append(neighbor_index)
        neighbor_indices_in_ground_truth_node_manager = np.array(neighbor_indices_in_ground_truth_node_manager).reshape(-1)
        # align the indices
        current_local_edge = neighbor_indices_in_ground_truth_node_manager
        current_in_edge = np.argwhere(current_local_edge == self.ground_truth_current_index)[0][0]
        current_local_edge = torch.tensor(current_local_edge).unsqueeze(0)
        
        k_size = current_local_edge.size()[-1]
        padding = torch.nn.ConstantPad1d((0, LOCAL_K_SIZE - k_size), 0)
        current_local_edge = padding(current_local_edge)
        
        current_local_edge = current_local_edge.unsqueeze(-1)
        local_edge_padding_mask = torch.zeros((1, 1, k_size), dtype=torch.int16).to(self.device)
        local_edge_padding_mask[0, 0, current_in_edge] = 1
        padding = torch.nn.ConstantPad1d((0, LOCAL_K_SIZE - k_size), 1)
        local_edge_padding_mask = padding(local_edge_padding_mask)
        
        return [local_node_inputs, local_node_padding_mask, local_edge_mask, current_local_index, current_local_edge, local_edge_padding_mask]

    def select_next_waypoint(self, local_observation, msg_stacked):
        _, _, _, _, current_local_edge, _ = local_observation
        with torch.no_grad():
            local_node_inputs, local_node_padding_mask, local_edge_mask, current_local_index, current_local_edge, local_edge_padding_mask = local_observation
            current_coord = torch.tensor(self.location, dtype=torch.float32).reshape(1, 1, 2).to(self.device)
            logp = self.policy_net(local_node_inputs, local_node_padding_mask, local_edge_mask, current_local_index, current_local_edge, local_edge_padding_mask, current_coord, msg_stacked)

        action_index = torch.multinomial(logp.exp(), 1).long().squeeze(1)
        next_node_index = current_local_edge[0, action_index.item(), 0].item()
        next_position = self.local_node_coords[next_node_index]

        return next_position, next_node_index, action_index
           
    def get_local_map(self, location):
        local_map_origin_x = (location[
                                  0] - self.local_map_size / 2) // self.downsampled_cell_size * self.downsampled_cell_size
        local_map_origin_y = (location[
                                  1] - self.local_map_size / 2) // self.downsampled_cell_size * self.downsampled_cell_size
        local_map_top_x = local_map_origin_x + self.local_map_size + NODE_RESOLUTION
        local_map_top_y = local_map_origin_y + self.local_map_size + NODE_RESOLUTION

        min_x = self.global_map_info.map_origin_x
        min_y = self.global_map_info.map_origin_y
        max_x = self.global_map_info.map_origin_x + self.cell_size * self.global_map_info.map.shape[1]
        max_y = self.global_map_info.map_origin_y + self.cell_size * self.global_map_info.map.shape[0]

        if local_map_origin_x < min_x:
            local_map_origin_x = min_x
        if local_map_origin_y < min_y:
            local_map_origin_y = min_y
        if local_map_top_x > max_x:
            local_map_top_x = max_x
        if local_map_top_y > max_y:
            local_map_top_y = max_y

        local_map_origin_x = np.around(local_map_origin_x, 1)
        local_map_origin_y = np.around(local_map_origin_y, 1)
        local_map_top_x = np.around(local_map_top_x, 1)
        local_map_top_y = np.around(local_map_top_y, 1)

        local_map_origin = np.array([local_map_origin_x, local_map_origin_y])
        local_map_origin_in_global_map = get_cell_position_from_coords(local_map_origin, self.global_map_info)

        local_map_top = np.array([local_map_top_x, local_map_top_y])
        local_map_top_in_global_map = get_cell_position_from_coords(local_map_top, self.global_map_info)

        local_map = self.global_map_info.map[
                    local_map_origin_in_global_map[1]:local_map_top_in_global_map[1],
                    local_map_origin_in_global_map[0]:local_map_top_in_global_map[0]]

        local_map_info = Map_info(local_map, local_map_origin_x, local_map_origin_y, self.cell_size)

        return local_map_info

    def get_extended_local_map(self, location):
        local_map_origin_x = (location[
                                  0] - self.extended_local_map_size / 2) // self.downsampled_cell_size * self.downsampled_cell_size
        local_map_origin_y = (location[
                                  1] - self.extended_local_map_size / 2) // self.downsampled_cell_size * self.downsampled_cell_size
        local_map_top_x = local_map_origin_x + self.extended_local_map_size + 2 * NODE_RESOLUTION
        local_map_top_y = local_map_origin_y + self.extended_local_map_size + 2 * NODE_RESOLUTION

        min_x = self.global_map_info.map_origin_x
        min_y = self.global_map_info.map_origin_y
        max_x = self.global_map_info.map_origin_x + self.cell_size * self.global_map_info.map.shape[1]
        max_y = self.global_map_info.map_origin_y + self.cell_size * self.global_map_info.map.shape[0]

        if local_map_origin_x < min_x:
            local_map_origin_x = min_x
        if local_map_origin_y < min_y:
            local_map_origin_y = min_y
        if local_map_top_x > max_x:
            local_map_top_x = max_x
        if local_map_top_y > max_y:
            local_map_top_y = max_y

        local_map_origin_x = np.around(local_map_origin_x, 1)
        local_map_origin_y = np.around(local_map_origin_y, 1)
        local_map_top_x = np.around(local_map_top_x, 1)
        local_map_top_y = np.around(local_map_top_y, 1)

        local_map_origin = np.array([local_map_origin_x, local_map_origin_y])
        local_map_origin_in_global_map = get_cell_position_from_coords(local_map_origin, self.global_map_info)

        local_map_top = np.array([local_map_top_x, local_map_top_y])
        local_map_top_in_global_map = get_cell_position_from_coords(local_map_top, self.global_map_info)

        local_map = self.global_map_info.map[
                    local_map_origin_in_global_map[1]:local_map_top_in_global_map[1],
                    local_map_origin_in_global_map[0]:local_map_top_in_global_map[0]]

        local_map_info = Map_info(local_map, local_map_origin_x, local_map_origin_y, self.cell_size)

        return local_map_info

    def save_observation(self, local_observation, stacked_msg):
        local_node_inputs, local_node_padding_mask, local_edge_mask, current_local_index, current_local_edge, local_edge_padding_mask = local_observation
        self.episode_buffer[0] += local_node_inputs
        self.episode_buffer[1] += local_node_padding_mask.bool()
        self.episode_buffer[2] += local_edge_mask.bool()
        self.episode_buffer[3] += current_local_index
        self.episode_buffer[4] += current_local_edge
        self.episode_buffer[5] += local_edge_padding_mask.bool()
        self.episode_buffer[6] += torch.tensor(self.location, dtype=torch.float32).reshape(1, 1, 2).to(self.device)
        self.episode_buffer[7] += stacked_msg

    def save_action(self, action_index):
        self.episode_buffer[8] += action_index.reshape(1, 1, 1)

    def save_reward(self, reward):
        self.episode_buffer[9] += torch.FloatTensor([reward]).reshape(1, 1, 1).to(self.device)

    def save_done(self, done):
        self.episode_buffer[10] += torch.tensor([int(done)]).reshape(1, 1, 1).to(self.device)

    def save_next_observations(self, local_observation, stacked_msg):
        self.episode_buffer[11] = copy.deepcopy(self.episode_buffer[0])[1:]
        self.episode_buffer[12] = copy.deepcopy(self.episode_buffer[1])[1:]
        self.episode_buffer[13] = copy.deepcopy(self.episode_buffer[2])[1:]
        self.episode_buffer[14] = copy.deepcopy(self.episode_buffer[3])[1:]
        self.episode_buffer[15] = copy.deepcopy(self.episode_buffer[4])[1:]
        self.episode_buffer[16] = copy.deepcopy(self.episode_buffer[5])[1:]
        self.episode_buffer[17] = copy.deepcopy(self.episode_buffer[6])[1:]
        self.episode_buffer[18] = [tensor.clone() for tensor in self.episode_buffer[7]][1:]

        local_node_inputs, local_node_padding_mask, local_edge_mask, current_local_index, current_local_edge, local_edge_padding_mask = local_observation
        self.episode_buffer[11] += local_node_inputs
        self.episode_buffer[12] += local_node_padding_mask.bool()
        self.episode_buffer[13] += local_edge_mask.bool()
        self.episode_buffer[14] += current_local_index
        self.episode_buffer[15] += current_local_edge
        self.episode_buffer[16] += local_edge_padding_mask.bool()
        self.episode_buffer[17] += torch.tensor(self.location, dtype=torch.float32).reshape(1, 1, 2).to(self.device)
        self.episode_buffer[18] += stacked_msg

    def save_ground_truth_observation(self, ground_truth_observation):
        local_node_inputs, local_node_padding_mask, local_edge_mask, current_local_index, current_local_edge, local_edge_padding_mask = ground_truth_observation
        self.ground_truth_episode_buffer[0] += local_node_inputs
        self.ground_truth_episode_buffer[1] += local_node_padding_mask.bool()
        self.ground_truth_episode_buffer[2] += local_edge_mask.bool()
        self.ground_truth_episode_buffer[3] += current_local_index
        self.ground_truth_episode_buffer[4] += current_local_edge
        self.ground_truth_episode_buffer[5] += local_edge_padding_mask.bool()
        
    
    def save_ground_truth_observations(self, ground_truth_observation):
        self.ground_truth_episode_buffer[6] = copy.deepcopy(self.ground_truth_episode_buffer[0])[1:]
        self.ground_truth_episode_buffer[7] = copy.deepcopy(self.ground_truth_episode_buffer[1])[1:]
        self.ground_truth_episode_buffer[8] = copy.deepcopy(self.ground_truth_episode_buffer[2])[1:]
        self.ground_truth_episode_buffer[9] = copy.deepcopy(self.ground_truth_episode_buffer[3])[1:]
        self.ground_truth_episode_buffer[10] = copy.deepcopy(self.ground_truth_episode_buffer[4])[1:]
        self.ground_truth_episode_buffer[11] = copy.deepcopy(self.ground_truth_episode_buffer[5])[1:]

        local_node_inputs, local_node_padding_mask, local_edge_mask, current_local_index, current_local_edge, local_edge_padding_mask = ground_truth_observation
        self.ground_truth_episode_buffer[6] += local_node_inputs
        self.ground_truth_episode_buffer[7] += local_node_padding_mask.bool()
        self.ground_truth_episode_buffer[8] += local_edge_mask.bool()
        self.ground_truth_episode_buffer[9] += current_local_index
        self.ground_truth_episode_buffer[10] += current_local_edge
        self.ground_truth_episode_buffer[11] += local_edge_padding_mask.bool()
        
        

    def get_no_padding_observation(self):
        local_node_coords = self.local_node_coords
        local_node_utility = self.utility.reshape(-1, 1)
        local_node_guidepost = self.guidepost.reshape(-1, 1)
        local_node_occupancy = self.occupancy.reshape(-1, 1)
        current_local_index = self.current_local_index
        local_edge_mask = self.local_adjacent_matrix
        current_local_edge = self.local_neighbor_indices
        n_local_node = local_node_coords.shape[0]

        current_local_node_coords = local_node_coords[self.current_local_index]
        local_node_coords = np.concatenate((local_node_coords[:, 0].reshape(-1, 1) - current_local_node_coords[0],
                                            local_node_coords[:, 1].reshape(-1, 1) - current_local_node_coords[1]),
                                            axis=-1) / LOCAL_MAP_SIZE
        local_node_utility = local_node_utility
        local_node_inputs = np.concatenate((local_node_coords, local_node_utility, local_node_guidepost,local_node_occupancy), axis=1)
        local_node_inputs = torch.FloatTensor(local_node_inputs).unsqueeze(0).to(self.device)

        
        local_node_padding_mask = torch.zeros((1, 1, n_local_node), dtype=torch.int16).to(self.device)

        current_local_index = torch.tensor([current_local_index]).reshape(1, 1, 1).to(self.device)

        local_edge_mask = torch.tensor(local_edge_mask).unsqueeze(0).to(self.device)

        current_in_edge = np.argwhere(current_local_edge == self.current_local_index)[0][0]
        current_local_edge = torch.tensor(current_local_edge).unsqueeze(0)
        k_size = current_local_edge.size()[-1]
        current_local_edge = current_local_edge.unsqueeze(-1)

        local_edge_padding_mask = torch.zeros((1, 1, k_size), dtype=torch.int16).to(self.device)
        local_edge_padding_mask[0, 0, current_in_edge] = 1

        return [local_node_inputs, local_node_padding_mask, local_edge_mask, current_local_index, current_local_edge, local_edge_padding_mask]

    def get_stacked_msg(self):
        stacked_msg = []
        for idx in range(N_AGENTS):
            # skip self
            if idx == self.id:
                continue
            msg = self.msgs[idx][-1]
            stacked_msg.append(msg)
        
        return torch.stack(stacked_msg, dim=1).squeeze(2)

    def get_self_msg(self):
        self_msg = self.msgs[self.id][-1]
        return self_msg

    def clear_newest_msg(self):
        print("clear newest msg")
        for msg in self.msgs:
            msg.pop()