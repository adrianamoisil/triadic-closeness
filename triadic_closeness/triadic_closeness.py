from typing import Optional, List, Dict, Tuple
from triadic_closeness.normalize import normalize_triadic_closeness
from triadic_closeness.weights import \
  compute_weights_using_number_of_neighbours, \
  compute_weights_using_number_of_common_features, \
  compute_weights_using_number_of_neighbours_and_common_features, \
  compute_weights_using_jaccard_similarity_for_neighbours, \
  compute_weights_using_jaccard_similarity_for_features
from ego_network import EgoNetwork


def get_triad_pattern(u: int, 
                      v: int, 
                      z: int, 
                      edges: List[Tuple[int, int]]) -> int:
  triad_pattern = 0

  # Check how the pairs (u, z) and (v, z) are directly connected to each other.
  if (z, u) in edges:
    if (z, v) in edges:
      if (u, z) in edges:
        if (v, z) in edges:
          triad_pattern = 1
        else:
          triad_pattern = 2
      else:
        if (v, z) in edges:
          triad_pattern = 7
        else:
          triad_pattern = 9
    else:
      if (u, z) in edges:
        if (v, z) in edges:
          triad_pattern = 5
      else:
        if (v, z) in edges:
          triad_pattern = 8
  elif (u, z) in edges:
    if (z, v) in edges:
      if (v, z) in edges:
        triad_pattern = 3
      else:
        triad_pattern = 4
    else:
      if (v, z) in edges:
        triad_pattern = 6

  # No pattern found between the given nodes.
  if triad_pattern == 0:
    return 0

  # Check how u and v are directly connected to each other
  if (u, v) in edges and \
    (v, u) in edges:
    triad_pattern = 30 + triad_pattern
  elif (u, v) in edges:
    triad_pattern = 10 + triad_pattern
  elif (v, u) in edges:
    triad_pattern = 20 + triad_pattern

  return triad_pattern


def compute_triad_patterns_frequencies(nodes: List[int], 
                                       edges: List[Tuple[int, int]]) -> Dict[int, int]:
  frequencies = {}

  for z in nodes:
    neighbours_of_z = [node for node in nodes \
                       if (node, z) in edges or (z, node) in edges]
    for u in neighbours_of_z:
      for v in neighbours_of_z:
        if u == v:
          continue
        triad_pattern = get_triad_pattern(u, v, z, edges)
        if triad_pattern:
          if triad_pattern in frequencies.keys():
            frequencies[triad_pattern] += 1
          else:
            frequencies[triad_pattern] = 1

  return frequencies


def compute_triadic_closeness(u: int, 
                              v: int,
                              nodes: List[int],
                              edges: List[Tuple[int, int]],
                              weights: Dict[int, float],
                              frequencies: Dict[int, int]) -> float:
  triadic_closeness = 0
  neighbours = [node for node in nodes \
    if node != u and node != v and \
      ((node, u) in edges or (u, node) in edges) and \
      ((node, v) in edges or (v, node) in edges)]
  for z in neighbours:
    triad_pattern = get_triad_pattern(u, v, z, edges)
    frequency_10 = frequencies[triad_pattern + 10] if triad_pattern + 10 in frequencies else 0
    frequency_30 = frequencies[triad_pattern + 30] if triad_pattern + 30 in frequencies else 0
    triadic_closeness += \
      ((frequency_10 + frequency_30)  * weights[z]) / frequencies[triad_pattern]

  return triadic_closeness


def _compute_triadic_closeness_for_all_nodes_v1(edges: list, 
                                                norm_triadic_closeness: Optional[str]) -> list:
  '''Weights are based on number of neighbours.'''
  nodes = sorted(set([node for edge in edges for node in edge]))
  weights = compute_weights_using_number_of_neighbours(nodes, edges)
  frequencies = compute_triad_patterns_frequencies(nodes, edges)
  results = []

  for first_node in nodes:
    for second_node in nodes:
      if first_node == second_node or \
        (first_node, second_node) in edges:
        continue
      triadic_closeness = compute_triadic_closeness(first_node, second_node, nodes, edges, weights, frequencies)
      if triadic_closeness:
        results.append((first_node, second_node, triadic_closeness))

  if norm_triadic_closeness is None:
    return results
  return normalize_triadic_closeness(results, norm_triadic_closeness)


def _compute_triadic_closeness_for_all_nodes_v2(ego_network: EgoNetwork,
                                                norm_triadic_closeness: Optional[str],
                                                norm_weights: Optional[str],
                                                use_preprocessed_features: bool) -> list:
  '''Weights are based on number of common features.'''
  nodes = sorted(set([node for edge in ego_network.edges for node in edge]))
  weights = \
    compute_weights_using_number_of_common_features(nodes, 
                                                    ego_network.preprocessed_features if use_preprocessed_features else ego_network.features, 
                                                    norm_weights)
  frequencies = compute_triad_patterns_frequencies(nodes, ego_network.edges)
  results = []

  for first_node in nodes:
    for second_node in nodes:
      if first_node == second_node or \
        (first_node, second_node) in ego_network.edges:
        continue
      triadic_closeness = compute_triadic_closeness(first_node, second_node, nodes, ego_network.edges, weights[first_node], frequencies)
      if triadic_closeness:
        results.append((first_node, second_node, triadic_closeness))

  if norm_triadic_closeness is None:
    return results
  return normalize_triadic_closeness(results, norm_triadic_closeness)


def _compute_triadic_closeness_for_all_nodes_v3(ego_network: EgoNetwork,
                                                norm_triadic_closeness: Optional[str],
                                                norm_weights: Optional[str],
                                                use_preprocessed_features: bool) -> list:
  '''Weights are based on number of common features and number of neighbours.'''
  nodes = sorted(set([node for edge in ego_network.edges for node in edge]))
  weights = \
    compute_weights_using_number_of_neighbours_and_common_features(nodes, 
                                                                   ego_network.edges, 
                                                                   ego_network.preprocessed_features if use_preprocessed_features else ego_network.features, 
                                                                   norm_weights)
  frequencies = compute_triad_patterns_frequencies(nodes, ego_network.edges)
  results = []

  for first_node in nodes:
    for second_node in nodes:
      if first_node == second_node or \
        (first_node, second_node) in ego_network.edges:
        continue
      triadic_closeness = compute_triadic_closeness(first_node, second_node, nodes, ego_network.edges, weights[first_node], frequencies)
      if triadic_closeness:
        results.append((first_node, second_node, triadic_closeness))

  if norm_triadic_closeness is None:
    return results
  return normalize_triadic_closeness(results, norm_triadic_closeness)


def _compute_triadic_closeness_for_all_nodes_v4(ego_network: EgoNetwork,
                                                norm_triadic_closeness: Optional[str]) -> list:
  '''Weights are based on Jaccard similarity.'''
  nodes = sorted(set([node for edge in ego_network.edges for node in edge]))
  weights = \
    compute_weights_using_jaccard_similarity_for_neighbours(nodes, ego_network.edges)
  frequencies = compute_triad_patterns_frequencies(nodes, ego_network.edges)
  results = []

  for first_node in nodes:
    for second_node in nodes:
      if first_node == second_node or \
        (first_node, second_node) in ego_network.edges:
        continue
      triadic_closeness = compute_triadic_closeness(first_node, second_node, nodes, ego_network.edges, weights[first_node], frequencies)
      if triadic_closeness:
        results.append((first_node, second_node, triadic_closeness))

  if norm_triadic_closeness is None:
    return results
  return normalize_triadic_closeness(results, norm_triadic_closeness)


def _compute_triadic_closeness_for_all_nodes_v5(ego_network: EgoNetwork,
                                                norm_triadic_closeness: Optional[str],
                                                use_preprocessed_features: bool) -> list:
  '''Weights are based on Jaccard similarity.'''
  nodes = sorted(set([node for edge in ego_network.edges for node in edge]))
  weights = \
    compute_weights_using_jaccard_similarity_for_features(nodes, 
                                                          ego_network.preprocessed_features if use_preprocessed_features else ego_network.features)
  frequencies = compute_triad_patterns_frequencies(nodes, ego_network.edges)
  results = []

  for first_node in nodes:
    for second_node in nodes:
      if first_node == second_node or \
        (first_node, second_node) in ego_network.edges:
        continue
      triadic_closeness = compute_triadic_closeness(first_node, second_node, nodes, ego_network.edges, weights[first_node], frequencies)
      if triadic_closeness:
        results.append((first_node, second_node, triadic_closeness))

  if norm_triadic_closeness is None:
    return results
  return normalize_triadic_closeness(results, norm_triadic_closeness)


def compute_triadic_closeness_for_all_nodes(ego_network: EgoNetwork,
                                            norm_triadic_closeness: Optional[str] = None,
                                            norm_weights: Optional[str] = None,
                                            weights_strategy: str = 'neighbours',
                                            use_preprocessed_features: bool = False) -> list:
  triadic_closeness = list()
  if weights_strategy == 'neighbours':
    triadic_closeness = _compute_triadic_closeness_for_all_nodes_v1(ego_network.edges, 
                                                                    norm_triadic_closeness)
  elif weights_strategy == 'features':
    triadic_closeness = _compute_triadic_closeness_for_all_nodes_v2(ego_network, 
                                                                    norm_triadic_closeness, 
                                                                    norm_weights,
                                                                    use_preprocessed_features=use_preprocessed_features)
  elif weights_strategy == 'neighbours_features':
    triadic_closeness = _compute_triadic_closeness_for_all_nodes_v3(ego_network, 
                                                                    norm_triadic_closeness, 
                                                                    norm_weights,
                                                                    use_preprocessed_features=use_preprocessed_features)
  elif weights_strategy == 'jaccard_neighbours':
    triadic_closeness = _compute_triadic_closeness_for_all_nodes_v4(ego_network, 
                                                                    norm_triadic_closeness)
  elif weights_strategy == 'jaccard_features':
    triadic_closeness = _compute_triadic_closeness_for_all_nodes_v5(ego_network, 
                                                                    norm_triadic_closeness,
                                                                    use_preprocessed_features=use_preprocessed_features)

  return sorted(triadic_closeness, reverse=True, key=lambda x: x[2])
