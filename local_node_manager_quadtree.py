import time

import numpy as np
from utils import *
from parameter import *
import quads


class Local_node_manager:
    def __init__(self, plot=False):
        self.local_nodes_dict = quads.QuadTree((0, 0), 1000, 1000)
        self.plot = plot
        if self.plot:
            self.x = []
            self.y = []

    def check_node_exist_in_dict(self, coords):
        key = (coords[0], coords[1])
        exist = self.local_nodes_dict.find(key)
        return exist

    def add_node_to_dict(self, coords, local_frontiers, extended_local_map_info):
        key = (coords[0], coords[1])
        node = Local_node(coords, local_frontiers, extended_local_map_info)
        self.local_nodes_dict.insert(point=key, data=node)

    def update_local_graph(self, robot_location, local_frontiers, local_map_info, extended_local_map_info):
        extended_local_node_coords, _ = get_local_node_coords(robot_location, extended_local_map_info)
        for coords in extended_local_node_coords:
            node = self.check_node_exist_in_dict(coords)
            if node is not None:
                node = node.data
                if node.utility == 0 or np.linalg.norm(node.coords - robot_location) > 2 * SENSOR_RANGE:
                    pass
                else:
                    node.update_node_observable_frontiers(local_frontiers, extended_local_map_info)

        local_node_coords, _ = get_local_node_coords(robot_location, local_map_info)

        for coords in local_node_coords:
            node = self.check_node_exist_in_dict(coords)
            if node is None:
                self.add_node_to_dict(coords, local_frontiers, extended_local_map_info)
            else:
                pass

        for coords in local_node_coords:
            node = self.local_nodes_dict.find((coords[0], coords[1])).data

            plot_x = self.x if self.plot else None
            plot_y = self.y if self.plot else None
            node.update_neighbor_nodes(extended_local_map_info, self.local_nodes_dict, plot_x, plot_y)

    def get_all_node_graph(self, robot_location, robot_locations):
        all_node_coords = []
        for node in self.local_nodes_dict.__iter__():
            all_node_coords.append(node.data.coords)
        all_node_coords = np.array(all_node_coords).reshape(-1, 2)
        utility = []

        n_nodes = all_node_coords.shape[0]
        adjacent_matrix = np.ones((n_nodes, n_nodes)).astype(int)
        local_node_coords_to_check = all_node_coords[:, 0] + all_node_coords[:, 1] * 1j
        for i, coords in enumerate(all_node_coords):
            node = self.local_nodes_dict.find((coords[0], coords[1])).data
            utility.append(node.utility)
            for neighbor in node.neighbor_list:
                index = np.argwhere(local_node_coords_to_check == neighbor[0] + neighbor[1] * 1j)
                if index or index == [[0]]:
                    index = index[0][0]
                    adjacent_matrix[i, index] = 0
        
        utility = np.array(utility)
        utility = utility / 30
        
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
                    node = self.local_nodes_dict.find(coords.tolist())
                    neighbor_boundary = self.get_neighbor_boundary(coords)
                    nodes_in_neighbor_range = self.local_nodes_dict.within_bb(neighbor_boundary)
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


        robot_in_graph = self.local_nodes_dict.nearest_neighbors(robot_location.tolist(), 1)[0].data.coords
        current_index = np.argwhere(local_node_coords_to_check == robot_in_graph[0] + robot_in_graph[1] * 1j)[0][0]
        neighbor_indices = np.argwhere(adjacent_matrix[current_index] == 0).reshape(-1)

        occupancy = np.zeros((n_nodes, 1))
        for location in robot_locations:
            try:
                location_in_graph = self.local_nodes_dict.find((location[0], location[1])).data.coords
                index = np.argwhere(local_node_coords_to_check == location_in_graph[0] + location_in_graph[1] * 1j)[0][0]
                if index == current_index:
                    occupancy[index] = -1
                else:
                    occupancy[index] = 1
            except:
                all_node_coords = np.concatenate((all_node_coords, location.reshape(1, 2)), axis=0)
                utility = np.concatenate((utility, -np.ones(1)), axis=0)
                guidepost = np.concatenate((guidepost, np.zeros(1)), axis=0)
                occupancy = np.concatenate((occupancy, np.ones((1,1))), axis=0)
                adjacent_matrix = np.concatenate((adjacent_matrix, np.zeros((1, n_nodes))), axis=0)
                adjacent_matrix = np.concatenate((adjacent_matrix, np.zeros((n_nodes + 1, 1))), axis=1)
                n_nodes += 1


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

    def a_star(self, start, destination, max_dist=1e8):
        # the path does not include the start
        if not self.check_node_exist_in_dict(start):
            Warning("start position is not in node dict")
            return [], 1e8
        if not self.check_node_exist_in_dict(destination):
            Warning("end position is not in node dict")
            return [], 1e8

        if start[0] == destination[0] and start[1] == destination[1]:
            return [destination], 0

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
                    node = self.local_nodes_dict.find(n).data
                    n_coords = node.coords
                    h_n = self.h(n_coords, destination)
                if n is None or g[v] + h_v < g[n] + h_n:
                    n = v
                    node = self.local_nodes_dict.find(n).data
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

        for node in self.local_nodes_dict.__iter__():
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

            node = self.local_nodes_dict.find(u).data
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


class Local_node:
    def __init__(self, coords, local_frontiers, extended_local_map_info):
        self.coords = coords
        self.utility_range = UTILITY_RANGE
        self.observable_frontiers = self.initialize_observable_frontiers(local_frontiers, extended_local_map_info)
        self.utility = self.observable_frontiers.shape[0] if self.observable_frontiers.shape[0] > MIN_UTILITY else 0
        self.utility_share = [self.utility]
        self.visited = 0

        self.neighbor_matrix = -np.ones((5, 5))
        self.neighbor_list = []
        self.neighbor_matrix[2, 2] = 1
        self.neighbor_list.append(self.coords)
        self.need_update_neighbor = True

    def initialize_observable_frontiers(self, local_frontiers, extended_local_map_info):
        if local_frontiers.shape[0] == 0:
            self.utility = 0
            return local_frontiers
        else:
            observable_frontiers = []
            dist_list = np.linalg.norm(local_frontiers - self.coords, axis=-1)
            frontiers_in_range = local_frontiers[dist_list < self.utility_range]
            for point in frontiers_in_range:
                collision = check_collision(self.coords, point, extended_local_map_info)
                if not collision:
                    observable_frontiers.append(point)
            observable_frontiers = np.array(observable_frontiers)
            return observable_frontiers

    def update_neighbor_nodes(self, extended_local_map_info, nodes_dict, plot_x=None, plot_y=None):
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
                        cell = get_cell_position_from_coords(neighbor_coords, extended_local_map_info)
                        if cell[0] < extended_local_map_info.map.shape[1] and cell[1] < extended_local_map_info.map.shape[0]:
                            if extended_local_map_info.map[cell[1], cell[0]] == 1:
                                self.neighbor_matrix[i, j] = 1
                            continue
                    else:
                        neighbor_node = neighbor_node.data
                        collision = check_collision(self.coords, neighbor_coords, extended_local_map_info)
                        neighbor_matrix_x = center_index + (center_index - i)
                        neighbor_matrix_y = center_index + (center_index - j)
                        if not collision:
                            self.neighbor_matrix[i, j] = 1
                            self.neighbor_list.append(neighbor_coords)

                            neighbor_node.neighbor_matrix[neighbor_matrix_x, neighbor_matrix_y] = 1
                            neighbor_node.neighbor_list.append(self.coords)

                            if plot_x is not None and plot_y is not None:
                                plot_x.append([self.coords[0], neighbor_coords[0]])
                                plot_y.append([self.coords[1], neighbor_coords[1]])

        if self.utility == 0:
            self.need_update_neighbor = False
        elif np.sum(self.neighbor_matrix) == self.neighbor_matrix.shape[0] ** 2:
            self.need_update_neighbor = False
        # print(self.neighbor_matrix)

    def update_node_observable_frontiers(self, local_frontiers, extended_local_map_info):

        # remove observed frontiers in the observable frontiers
        if local_frontiers.shape[0] == 0:
            self.utility = 0
            self.utility_share[0] = self.utility
            self.observable_frontiers = local_frontiers
            return

        local_frontiers = local_frontiers.reshape(-1, 2)
        old_frontier_to_check = self.observable_frontiers[:, 0] + self.observable_frontiers[:, 1] * 1j
        local_frontiers_to_check = local_frontiers[:, 0] + local_frontiers[:, 1] * 1j
        to_observe_index = np.where(
            np.isin(old_frontier_to_check, local_frontiers_to_check, assume_unique=True) == True)
        new_frontier_index = np.where(
            np.isin(local_frontiers_to_check, old_frontier_to_check, assume_unique=True) == False)
        self.observable_frontiers = self.observable_frontiers[to_observe_index]
        new_frontiers = local_frontiers[new_frontier_index]

        # add new frontiers in the observable frontiers
        if new_frontiers.shape[0] > 0:
            dist_list = np.linalg.norm(new_frontiers - self.coords, axis=-1)
            new_frontiers_in_range = new_frontiers[dist_list < self.utility_range]
            for point in new_frontiers_in_range:
                collision = check_collision(self.coords, point, extended_local_map_info)
                if not collision:
                    self.observable_frontiers = np.concatenate((self.observable_frontiers, point.reshape(1, 2)), axis=0)
        self.utility = self.observable_frontiers.shape[0]
        if self.utility <= MIN_UTILITY:
            self.utility = 0
        self.utility_share[0] = self.utility

    def delete_observed_frontiers(self, observed_frontiers):
        # remove observed frontiers in the observable frontiers
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
        self.visited = 1
        self.observable_frontiers = np.array([[], []]).reshape(0, 2)
        self.utility = 0
        self.utility_share[0] = self.utility