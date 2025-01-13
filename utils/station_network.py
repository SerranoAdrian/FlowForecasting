import pandas as pd
import networkx as nx
from pyvis.network import Network
import random
from itertools import chain, product
import geopy.distance
import numpy as np
from copy import deepcopy

class StationNetworkSimul:
    def __init__(self, df_stations : pd.DataFrame, df_pos : pd.DataFrame):

        self._init_network_dicts(df_stations=df_stations)
        self._init_network_graph(df_stations=df_stations)
        self._set_nodes_positions(df_pos=df_pos)
    
    def set_edges_weights(self):
        for edge in self.network_graph.edges:
            start_node = self.network_graph.nodes[edge[0]]
            end_node = self.network_graph.nodes[edge[1]]
            distance = geopy.distance.geodesic((start_node['lat'], start_node['lon']), (end_node['lat'], end_node['lon'])).km
            self.network_graph.edges[edge]['weight'] = 1/distance if distance != 0 else 1.0
    

    def set_nodes_traffic(self, df_flow):
        nodes_traffic = {node_idx : {'traffic' : 0} for node_idx in self.network_graph.nodes}
        # self.shortest_path_cache = pd.DataFrame(
        #     np.zeros((df_flow.shape[0],len(self.network_graph.edges))),
        #     columns=[edge for edge in self.network_graph.edges],
        #     dtype=int
        # )
        self.shortest_path_cache = {path_idx : [] for path_idx in range(len(df_flow))}
        self.shortest_path_cache_reverse = {edge : [] for edge in self.network_graph.edges}
        for path_idx in range(len(df_flow)):
            path = df_flow.iloc[path_idx]
            start_station = path['de']
            end_station = path['vers']
            flow = path['nombre']
            best_path = self.get_best_path(self.network_graph, start_station, end_station)
            if best_path is not None:
                for node in best_path:
                    nodes_traffic[node]['traffic'] += flow
                for s,t in nx.utils.pairwise(best_path):
                    self.shortest_path_cache[path_idx].append((s,t))
                    self.shortest_path_cache_reverse[(s,t)].append(path_idx)
                    # self.shortest_path_cache[path_idx] = [(s,t) for s,t in nx.utils.pairwise(best_path)]

                # for i in range(len(best_path)-1):
                #     # self.shortest_path_cache[(best_path[i], best_path[i+1])].append(path_idx)
                #     self.shortest_path_cache.iloc[path_idx][(best_path[i], best_path[i+1])] = 1
        
        # total_trafic = sum([node['traffic'] for node in nodes_traffic.values()])
        # normalized_nodes_traffic = {node_idx : {'traffic' : node['traffic']/total_trafic} for node_idx, node in nodes_traffic.items()}
        nx.set_node_attributes(self.network_graph, nodes_traffic)
    
    def set_edges_traffic(self, df_flow):
        edges_traffic = {edge : {'traffic' : 0} for edge in self.network_graph.edges}
        self.shortest_path_cache = {path_idx : [] for path_idx in range(len(df_flow))}
        self.shortest_path_cache_reverse = {edge : [] for edge in self.network_graph.edges}
        for path_idx in range(len(df_flow)):
            path = df_flow.iloc[path_idx]
            start_station = path['de']
            end_station = path['vers']
            flow = path['nombre']
            best_path = self.get_best_path(self.network_graph, start_station, end_station)
            if best_path is not None:
                for edge in nx.utils.pairwise(best_path):
                    edges_traffic[edge]['traffic']+=flow
                    self.shortest_path_cache[path_idx].append(edge)
                    self.shortest_path_cache_reverse[edge].append(path_idx)

        nx.set_edge_attributes(self.network_graph, edges_traffic)
    

    def get_best_path(self, G, start_station, end_station):
        lb_path_weight = 0
        best_path = None
        for start_node in self.network_stations[start_station].values():
            for end_node in self.network_stations[end_station].values():
                try:
                    current_path = nx.dijkstra_path(G, start_node, end_node, weight='weight')
                    current_path_weight = sum([G.edges[edge]['weight'] for edge in zip(current_path[:-1], current_path[1:])])
                    if current_path_weight > lb_path_weight:
                        lb_path_weight = current_path_weight
                        best_path = current_path
                except nx.NetworkXNoPath:
                    print(f"Could not find a path between {self.reverse_network_stations[start_node]['title']} (line {self.reverse_network_stations[start_node]['group']}) \
                        and {self.reverse_network_stations[end_node]['title']} (line {self.reverse_network_stations[end_node]['group']})")
        return best_path

    def get_degraded_network(self, num_delete=10):
        degraded_graph = self.network_graph.copy()
        graph_edges = deepcopy(list(self.network_graph.edges))
        removed_edges = []
        for i in range(num_delete):
            is_deletable = False
            while not is_deletable:
                deleted_edge = random.choice(graph_edges)
                simple_paths = [simple_path for simple_path in nx.all_simple_paths(degraded_graph, deleted_edge[0], deleted_edge[1], cutoff=5)]
                if len(simple_paths) > 1:
                    degraded_graph.remove_edge(deleted_edge[0], deleted_edge[1])
                    graph_edges.remove(deleted_edge)
                    is_deletable = True
                    removed_edges.append(deleted_edge)

        
        return degraded_graph, removed_edges
    
    def update_degraded_network_nodes_traffic(self, new_net, removed_edges, df_flow):
        affected_paths = set()
        for removed_edge in removed_edges:
            affected_paths = affected_paths.union(set(self.shortest_path_cache_reverse[removed_edge]))

        #updating traffic for all the affected pathes
        for path_idx in affected_paths:
            affected_edges = self.shortest_path_cache[path_idx]
            
            #removing traffic calculated for nodes in the initial path
            new_net.nodes[affected_edges[0][0]]['traffic']-=df_flow.iloc[path_idx]['nombre']
            for edge in affected_edges:
                new_net.nodes[edge[1]]['traffic']-=df_flow.iloc[path_idx]['nombre']
            
            #adding traffic for all nodes in the new path
            new_path = nx.dijkstra_path(new_net, affected_edges[0][0], affected_edges[-1][-1])
            for node in new_path:
                new_net.nodes[node]['traffic']+=df_flow.iloc[path_idx]['nombre']
    def update_degraded_network_edges_traffic(self, new_net, removed_edges, df_flow):
        affected_paths = set()
        for removed_edge in removed_edges:
            affected_paths = affected_paths.union(set(self.shortest_path_cache_reverse[removed_edge]))

        #updating traffic for all the affected pathes
        for path_idx in affected_paths:
            affected_edges = self.shortest_path_cache[path_idx]
            
            #removing traffic calculated for nodes in the initial path
            for edge in affected_edges:
                if new_net.has_edge(*edge):
                    new_net.edges[edge]['traffic']-=df_flow.iloc[path_idx]['nombre']
            
            new_path = nx.dijkstra_path(new_net, affected_edges[0][0], affected_edges[-1][-1])
            #adding traffic for all nodes in the new path
            for edge in nx.utils.pairwise(new_path):
                new_net.edges[edge]['traffic']+=df_flow.iloc[path_idx]['nombre']
    def _init_network_dicts(self, df_stations: pd.DataFrame):
        self.network_stations = {station : {} for station in df_stations['de Station'].unique()}
        self.reverse_network_stations = {}

        # unique (station, line) couples that we're using as nodes in our graph
        df_nodes = df_stations.groupby(['de Station','de Ligne']).size().reset_index().rename(columns={0:'count'})

        count=0
        for node in df_nodes.values:
            self.network_stations[node[0]].update({node[1] : count})
            self.reverse_network_stations.update({count : {'title' : node[0], 'group' :node[1]}})
            count+=1
    
    def _init_network_graph(self, df_stations: pd.DataFrame):
        edges = []
        for edge in df_stations.values:
            edges.append((self.network_stations[edge[1]][edge[0]], self.network_stations[edge[3]][edge[2]]))
        self.network_graph = nx.DiGraph()
        self.network_graph.add_edges_from(edges)
        nx.set_node_attributes(self.network_graph, self.reverse_network_stations)

        for station_lines in self.network_stations.values():
            station_nodes = station_lines.values()
            if len(station_nodes)>1:
                for edge in product(station_nodes, repeat=2):
                    if not self.network_graph.has_edge(*edge) and edge[0] != edge[1]:
                        self.network_graph.add_edge(*edge)

    def get_node_neighbors(self, node, distance=1):
        if distance == 1:
            neighbors = list(self.network_graph.neighbors(node))
            return neighbors
        else:
            neighbors = list(self.network_graph.neighbors(node))
            total_neighbors = [neighbors.copy()]
            for neighbor in neighbors:
                total_neighbors.append(self.get_node_neighbors(neighbor, distance-1))
            total_neighbors = list(chain(*total_neighbors))
            return list(set(total_neighbors))
    
    def _set_nodes_positions(self, df_pos : pd.DataFrame, min_dist=3, max_dist=3):
        
        unspecified_loc_nodes = []
        for station, station_lines in self.network_stations.items():
            df_coords = df_pos[df_pos['Station'] == station]
            if not df_coords.empty:
                complete_gps = []
                if len(station_lines.values()) - len(df_coords['GPS'].values) > 0:
                    for _ in range(len(station_lines.values()) - len(df_coords['GPS'].values)):
                        complete_gps.append(random.choice(df_coords['GPS'].values))

                for coord, node_station in zip([*df_coords['GPS'].values, *complete_gps], station_lines.values()):
                    lat, lon = coord.split(', ')
                    lat, lon = float(lat), float(lon)
                    self.reverse_network_stations[node_station].update({'lon':lon, 'lat':lat})
            else:
                unspecified_loc_nodes = [*unspecified_loc_nodes, *station_lines.values()]
                
        nx.set_node_attributes(self.network_graph, self.reverse_network_stations)

        unspecified_loc_nodes_pos = {}
        for node_station in unspecified_loc_nodes:
            mean_lat, mean_lon = [], []
            for dist in range(min_dist,max_dist+1):
                for neighbor in self.get_node_neighbors(node_station, distance=dist):

                    lon = self.reverse_network_stations[neighbor].get('lon')
                    lat = self.reverse_network_stations[neighbor].get('lat')
                    if lon is not None and lat is not None:
                        mean_lat.append(lat)
                        mean_lon.append(lon)

                if len(mean_lat) > 0 and len(mean_lon) > 0:
                    unspecified_loc_nodes_pos.update({node_station : {'lon' : sum(mean_lon)/len(mean_lon), 'lat': sum(mean_lat)/len(mean_lat)}})
                    break
                else:
                    print(node_station)
                    continue
        nx.set_node_attributes(self.network_graph, unspecified_loc_nodes_pos)


    
    
    
    

