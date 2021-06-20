from collections import OrderedDict, defaultdict
from typing import Callable, Tuple, Dict, List

import numpy as np
from tqdm.auto import tqdm


def distance(pointA: np.ndarray, documents: np.ndarray) -> np.ndarray:
    return np.linalg.norm(pointA - documents, axis=1, keepdims=True) 


def create_sw_graph(
        data: np.ndarray,
        num_candidates_for_choice_long: int = 10,
        num_edges_long: int = 5,
        num_candidates_for_choice_short: int = 10,
        num_edges_short: int = 5,
        use_sampling: bool = False,
        sampling_share: float = 0.05,
        dist_f: Callable = distance
    ) -> Dict[int, List[int]]:
    
    N = data.shape[0]
    if use_sampling:
        sample_N = int(N * sampling_share)
        all_docs = list(range(N))
    
    edges = dict()
    for i in range(N):
        
        if use_sampling:
            indexes = np.random.choice(all_docs, sample_N , replace=False)
        
        distances = dist_f(data[i], data).squeeze()
        
        sorted_indexes = np.argsort(distances)
        if not use_sampling:
            sorted_indexes = sorted_indexes[1:]
        elif sorted_indexes[0] == i:
            sorted_indexes = sorted_indexes[1:]
        
        short_candidates = sorted_indexes[:num_candidates_for_choice_short]        
        long_candidates = sorted_indexes[-num_candidates_for_choice_long:]

        long_edges = np.random.choice(long_candidates, num_edges_long, replace=False)
        short_edges = np.random.choice(short_candidates, num_edges_short, replace=False)
        

        edges[i] = []
        for el in short_edges:
            edges[i].append(el)
        for el in long_edges:
            edges[i].append(el)
        
    return edges

def control_queue(queue: list, visited_vertex: dict, search_k: int, edges: list):
    if queue and len(visited_vertex) < search_k:
        remainder = list(set(edges).difference(set(visited_vertex.keys())))
        queue.append(np.random.choice(remainder, 1)[0])

def nsw(query_point: np.ndarray, all_documents: np.ndarray, 
        graph_edges: Dict[int, List[int]],
        search_k: int = 10, num_start_points: int = 5,
        dist_f: Callable = distance) -> np.ndarray:
    
    edges = list(range(all_documents.shape[0]))
    
    queue = list(np.random.choice(edges, num_start_points, replace=False))
    visited_vertex = dict()
    
    while len(queue) > 0:
        point = queue.pop()
        if point in visited_vertex:
            control_queue(queue, visited_vertex, search_k, edges)
            continue
        else:
            neighbours = []
            for neighbour in graph_edges[point]:
                if neighbour in visited_vertex:
                    continue
                neighbours.append(neighbour)
            distances = dist_f(query_point, all_documents[neighbours]).squeeze()
            if len(neighbours) == 1:
                distances = [distances]
            visited_vertex.update(list(zip(neighbours, distances)))
            queue.extend(neighbours)
        control_queue(queue, visited_vertex, search_k, edges)
        
    nearest = list(zip(*sorted(visited_vertex.items(), key=lambda x: x[1])))[0][:search_k]
    return nearest


