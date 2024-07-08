import numpy as np
import quads
from copy import deepcopy
import time

from parameter import *
from utils import *

class Ground_truth_node_manager:
    def __init__(self, ground_truth_map_info, plot=False):
        self.ground_truth_nodes_dict = quads.QuadTree((0, 0), 1000, 1000)
        self.ground_truth_map_info = ground_truth_map_info
        self.ground_truth_map_info.map = ((self.ground_truth_map_info.map == 255) * 128) + 127
        # self.ground_truth_frontiers = get_frontier_in_map(self.ground_truth_map_info)
        self.free_area = get_free_area_coords(self.ground_truth_map_info)
        
        self.ground_truth_node_coords, self.utility = self.initial_ground_truth_graph()
        
        self.plot = plot
        if self.plot:
            self.x = []
            self.y = []

    def add_node_to_dict(self, coords):
        key = (coords[0], coords[1])
        node = Ground_truth_node(coords, self.free_area, self.ground_truth_map_info)
        self.ground_truth_nodes_dict.insert(point=key, data=node)

    def initial_ground_truth_graph(self):
        ground_truth_node_coords = get_ground_truth_node_coords(self.ground_truth_map_info)
        for coords in ground_truth_node_coords:
            self.add_node_to_dict(coords)
        for coords in ground_truth_node_coords:
            node = self.ground_truth_nodes_dict.find((coords[0], coords[1])).data
            node.update_neighbor_nodes(self.ground_truth_map_info, self.ground_truth_nodes_dict)

        utility = []
        for coords in ground_truth_node_coords:
            node = self.ground_truth_nodes_dict.find((coords[0], coords[1])).data
            utility.append(node.utility)
        utility = np.array(utility)
        return ground_truth_node_coords, utility

    def update_ground_truth_graph(self, belief_map_info):
        observed_obstacles_map = belief_map_info.map
        updated_map = self.ground_truth_map_info.map
        updated_map = np.where(observed_obstacles_map == 1, observed_obstacles_map, updated_map)
        self.ground_truth_map_info.map = updated_map
        
        # update explored free space
        free_space = np.where(belief_map_info.map == 255)
        # change the explored free space to 100 in the ground truth map
        self.ground_truth_map_info.map[free_space] = 100

        self.free_area = get_free_area_coords(self.ground_truth_map_info)
    
        

        for node in self.ground_truth_nodes_dict.__iter__():
            if node.data.utility > 0:
                node.data.update_node_observable_free(self.free_area)

        utility = []
        for coords in self.ground_truth_node_coords:
            node = self.ground_truth_nodes_dict.find((coords[0], coords[1])).data
            utility.append(node.utility)
        self.utility = np.array(utility)
        
    def get_all_node_graph(self, robot_location, robot_locations):
        all_node_coords = []
        for node in self.ground_truth_nodes_dict.__iter__():
            all_node_coords.append(node.data.coords)
        all_node_coords = np.array(all_node_coords).reshape(-1, 2)
        utility = []
        
        n_nodes = all_node_coords.shape[0]
        adjacent_matrix = np.ones((n_nodes, n_nodes)).astype(int)
        local_node_coords_to_check = all_node_coords[:, 0] + all_node_coords[:, 1] * 1j
        for i, coords in enumerate(all_node_coords):
            node = self.ground_truth_nodes_dict.find((coords[0], coords[1])).data
            utility.append(node.utility)
            for neighbor in node.neighbor_list:
                index = np.argwhere(local_node_coords_to_check == neighbor[0] + neighbor[1] * 1j)
                if index or index == [[0]]:
                    index = index[0][0]
                    adjacent_matrix[i, index] = 0

        utility = np.array(utility)
        utility = utility / 1000

        indices = np.argwhere(utility > 0).reshape(-1)
        utility_node_coords = all_node_coords[indices]
        utility_node_to_check = all_node_coords[:, 0] + all_node_coords[:, 1] * 1j
        dist_dict, prev_dict = self.Dijkstra(robot_location)
        if utility_node_coords.shape[0] > 5:
            key_center_indices = []
            coverd_center_indices = []
            for i, coords in enumerate(utility_node_coords):
                if i in coverd_center_indices:
                    pass
                elif np.linalg.norm(coords - robot_location) < 20:
                    key_center_indices.append(i)
                else:
                    key_center_indices.append(i)
                    node = self.ground_truth_nodes_dict.find(coords.tolist())
                    neighbor_boundary = self.get_neighbor_boundary(coords)
                    nodes_in_neighbor_range = self.ground_truth_nodes_dict.within_bb(neighbor_boundary)
                    if nodes_in_neighbor_range:
                        for neighbor in nodes_in_neighbor_range:
                            neighbor_coords = neighbor.data.coords
                            utility_neighbor = np.argwhere(utility_node_to_check == neighbor_coords[0] + neighbor_coords[1] * 1j)
                            if utility_neighbor:
                                coverd_center_indices.append(utility_neighbor[0])
            utility_node_coords = utility_node_coords[key_center_indices]

        guidepost = np.zeros_like(utility)
        for end in utility_node_coords:
            path_coords, _ = self.get_Dijkstra_path_and_dist(dist_dict, prev_dict, end)
            for coords in path_coords:
                index = np.argwhere(all_node_coords[:, 0] + all_node_coords[:, 1] * 1j == coords[0] + coords[1] * 1j)[0]
                guidepost[index] = 1


        robot_in_graph = self.ground_truth_nodes_dict.nearest_neighbors(robot_location.tolist(), 1)[0].data.coords
        current_index = np.argwhere(local_node_coords_to_check == robot_in_graph[0] + robot_in_graph[1] * 1j)[0][0]
        neighbor_indices = np.argwhere(adjacent_matrix[current_index] == 0).reshape(-1)

        occupancy = np.zeros((n_nodes, 1))
        for location in robot_locations:
            location_in_graph = self.ground_truth_nodes_dict.find((location[0], location[1])).data.coords
            index = np.argwhere(local_node_coords_to_check == location_in_graph[0] + location_in_graph[1] * 1j)[0][0]
            if index == current_index:
                occupancy[index] = -1
            else:
                occupancy[index] = 1
        return all_node_coords, utility, guidepost, occupancy, adjacent_matrix, current_index, neighbor_indices
    
    def get_neighbor_boundary(self, coords):
        min_x = coords[0] - 20
        min_y = coords[1] - 20
        max_x = coords[0] + 20
        max_y = coords[1] + 20
        neighbor_boundary = quads.BoundingBox(min_x, min_y, max_x, max_y)
        return neighbor_boundary

    def h(self, coords_1, coords_2):
        h = ((coords_1[0] - coords_2[0]) ** 2 + (coords_1[1] - coords_2[1]) ** 2) ** (1 / 2)
        h = np.round(h, 2)
        return h

    def check_node_exist_in_dict(self, coords):
        key = (coords[0], coords[1])
        exist = self.ground_truth_nodes_dict.find(key)
        return exist

    def a_star(self, start, destination, max_dist=1e8):
        if not self.check_node_exist_in_dict(start):
            print('start does not existed')
            return [], 1e8
        if not self.check_node_exist_in_dict(destination):
            print('destination does not existed')
            return [], 1e8
        if start[0] == destination[0] and start[1] == destination[1]:
            return [], 0

        open_list = {(start[0], start[1])}
        closed_list = set()
        g = {(start[0], start[1]): 0}
        parents = {(start[0], start[1]): (start[0], start[1])}

        while len(open_list) > 0:
            n = None
            h_n = 1e8

            for v in open_list:
                h_v = self.h(v, destination)
                if n is not None:
                    node = self.ground_truth_nodes_dict.find(n).data
                    n_coords = node.coords
                    h_n = self.h(n_coords, destination)
                if n is None or g[v] + h_v < g[n] + h_n:
                    n = v
                    node = self.ground_truth_nodes_dict.find(n).data
                    n_coords = node.coords

            if n_coords[0] == destination[0] and n_coords[1] == destination[1]:
                path = []
                length = g[n]
                while parents[n] != n:
                    path.append(n)
                    n = parents[n]
                path.reverse()
                return path, np.round(length, 2)

            for neighbor_node_coords in node.neighbor_list:
                cost = ((neighbor_node_coords[0] - n_coords[0]) ** 2 + (
                            neighbor_node_coords[1] - n_coords[1]) ** 2) ** (1 / 2)
                cost = np.round(cost, 2)
                m = (neighbor_node_coords[0], neighbor_node_coords[1])
                if g[n] + cost > max_dist:
                    continue
                if m not in open_list and m not in closed_list:
                    open_list.add(m)
                    parents[m] = n
                    g[m] = g[n] + cost
                else:
                    if g[m] > g[n] + cost:
                        g[m] = g[n] + cost
                        parents[m] = n

                        if m in closed_list:
                            closed_list.remove(m)
                            open_list.add(m)
            open_list.remove(n)
            closed_list.add(n)
        print('Path does not exist!')

        return [], 1e8

    def Dijkstra(self, start):
        q = set()
        dist_dict = {}
        prev_dict = {}

        for node in self.ground_truth_nodes_dict.__iter__():
            coords = node.data.coords
            key = (coords[0], coords[1])
            dist_dict[key] = 1e8
            prev_dict[key] = None
            q.add(key)

        dist_dict[(start[0], start[1])] = 0

        while len(q) > 0:

            u = None
            for coords in q:
                if u is None:
                    u = coords
                elif dist_dict[coords] < dist_dict[u]:
                    u = coords

            q.remove(u)

            node = self.ground_truth_nodes_dict.find(u).data
            for neighbor_node_coords in node.neighbor_list:
                v = (neighbor_node_coords[0], neighbor_node_coords[1])
                if v in q:
                    cost = ((neighbor_node_coords[0] - u[0]) ** 2 + (
                            neighbor_node_coords[1] - u[1]) ** 2) ** (1 / 2)
                    cost = np.round(cost, 2)
                    alt = dist_dict[u] + cost
                    if alt < dist_dict[v]:
                        dist_dict[v] = alt
                        prev_dict[v] = u

        return dist_dict, prev_dict

    def get_Dijkstra_path_and_dist(self, dist_dict, prev_dict, end):
        dist = dist_dict[(end[0], end[1])]

        path = [(end[0], end[1])]
        prev_node = prev_dict[(end[0], end[1])]
        while prev_node is not None:
            path.append(prev_node)
            temp = prev_node
            prev_node = prev_dict[temp]

        path.reverse()
        return path[1:], np.round(dist, 2)


class Ground_truth_node:
    def __init__(self, coords, free_area, ground_truth_map_info):
        self.coords = coords
        self.utility_range = UTILITY_RANGE
        self.observable_free = self.initialize_observable_free(free_area, ground_truth_map_info)
        self.utility = self.observable_free.shape[0] if self.observable_free.shape[0] > MIN_UTILITY else 0

        self.neighbor_matrix = -np.ones((5, 5))
        self.neighbor_list = []
        self.neighbor_matrix[2, 2] = 1
        self.neighbor_list.append(self.coords)
        
    def initialize_observable_free(self, free_area, ground_truth_map_info):
        if free_area.shape[0] == 0:
            self.utility = 0
            return free_area
        else:
            observable_free = []
            dist_list = np.linalg.norm(free_area - self.coords, axis=-1)
            free_area_in_range = free_area[dist_list < self.utility_range]
            for point in free_area_in_range:
                collision = check_collision(self.coords, point, ground_truth_map_info)
                if not collision:
                    observable_free.append(point)
            observable_free = np.array(observable_free)
            return observable_free
        

    def update_neighbor_nodes(self, ground_truth_map_info, nodes_dict):
        for i in range(self.neighbor_matrix.shape[0]):
            for j in range(self.neighbor_matrix.shape[1]):
                if self.neighbor_matrix[i, j] != -1:
                    continue
                else:
                    center_index = self.neighbor_matrix.shape[0] // 2
                    if i == center_index and j == center_index:
                        self.neighbor_matrix[i, j] = 1
                        continue

                    neighbor_coords = np.around(np.array([self.coords[0] + (i - center_index) * NODE_RESOLUTION,
                                                          self.coords[1] + (j - center_index) * NODE_RESOLUTION]), 1)
                    neighbor_node = nodes_dict.find((neighbor_coords[0], neighbor_coords[1]))
                    if neighbor_node is None:
                        cell = get_cell_position_from_coords(neighbor_coords, ground_truth_map_info)
                        if cell[0] < ground_truth_map_info.map.shape[1] and cell[1] < ground_truth_map_info.map.shape[0]:
                            if ground_truth_map_info.map[cell[1], cell[0]] == 1:
                                self.neighbor_matrix[i, j] = 1
                            continue
                    else:
                        neighbor_node = neighbor_node.data
                        collision = check_collision(self.coords, neighbor_coords, ground_truth_map_info)
                        neighbor_matrix_x = center_index + (center_index - i)
                        neighbor_matrix_y = center_index + (center_index - j)
                        if not collision:
                            self.neighbor_matrix[i, j] = 1
                            self.neighbor_list.append(neighbor_coords)

                            neighbor_node.neighbor_matrix[neighbor_matrix_x, neighbor_matrix_y] = 1
                            neighbor_node.neighbor_list.append(self.coords)
            
    def update_node_observable_free(self, free_area):
        free_area = free_area.reshape(-1, 2)
        old_free_to_check = self.observable_free[:, 0] + self.observable_free[:, 1] * 1j
        local_free_to_check = free_area[:, 0] + free_area[:, 1] * 1j
        to_observe_index = np.where(
            np.isin(old_free_to_check, local_free_to_check, assume_unique=True) == True)
        self.observable_free = self.observable_free[to_observe_index]
        self.utility = self.observable_free.shape[0] if self.observable_free.shape[0] > MIN_UTILITY else 0

    def delete_observed_frontiers(self, observed_frontiers):
        observed_frontiers = observed_frontiers.reshape(-1, 2)
        old_frontier_to_check = self.observable_frontiers[:, 0] + self.observable_frontiers[:, 1] * 1j
        observed_frontiers_to_check = observed_frontiers[:, 0] + observed_frontiers[:, 1] * 1j
        to_observe_index = np.where(
            np.isin(old_frontier_to_check, observed_frontiers_to_check, assume_unique=True) == False)
        self.observable_frontiers = self.observable_frontiers[to_observe_index]

        self.utility = self.observable_frontiers.shape[0]
        if self.utility <= MIN_UTILITY:
            self.utility = 0

    def set_visited(self):
        self.observable_frontiers = np.array([[], []]).reshape(0, 2)
        self.utility = 0