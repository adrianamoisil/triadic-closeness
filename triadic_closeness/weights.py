from typing import Optional, Dict, List, Tuple
from sklearn.preprocessing import normalize


# TC_n
def compute_weights_using_number_of_neighbours(nodes: List[int], 
                                               edges: List[Tuple[int, int]]) -> Dict[int, float]:
  '''1/number_of_neighbours
  
  Output:
    - dict of size len(nodes)'''
  weights = {}
  for z in nodes:
    number_of_neighbours = 0
    for other_node in nodes:
      if z == other_node:
        continue
      if (z, other_node) in edges or (other_node, z) in edges:
        number_of_neighbours += 1
    weight = 1 / number_of_neighbours
    weights[z] = weight
  return weights


# TC_f, TC_fp
def compute_weights_using_number_of_common_features(nodes: List[int], 
                                                    features: list,
                                                    norm: Optional[str]) -> Dict[int, Dict[int, float]]:
  '''number_of_common_features
  
  Output:
    - n x n dict, where n = len(nodes)
  '''
  weights = {}
  for u in nodes:
    weights_for_node = {}
    for z in nodes:
      if u == z:
        continue
      number_of_common_features = 0
      for feature in features[u].keys():
        number_of_common_features += (features[u][feature] & features[z][feature])
      weights_for_node[z] = number_of_common_features
    weights[u] = weights_for_node
  
  if norm is None:
    return weights
  
  all_weights = [weight for weights_for_node in weights.values() for weight in weights_for_node.values()]
  all_weights = sorted(list(dict.fromkeys(all_weights)))
  all_weights_normalized = normalize([all_weights], norm=norm)[0]
  normalization_mapping = dict([(all_weights[index], all_weights_normalized[index]) for index in range(len(all_weights))])

  normalized_weights = {}
  for first_node in weights.keys():
    normalized_weights_for_node = {}
    for second_node in weights[first_node].keys():
      normalized_weights_for_node[second_node] = \
        normalization_mapping[weights[first_node][second_node]]
    normalized_weights[first_node] = normalized_weights_for_node
  
  return normalized_weights


# TC_nf, TC_nfp
def compute_weights_using_number_of_neighbours_and_common_features(nodes: List[int], 
                                                                   edges: List[Tuple[int, int]],
                                                                   features: list,
                                                                   norm: Optional[str]) -> Dict[int, Dict[int, float]]:
  '''number_of_common_features/number_of_neighbours
  
  Output:
    - n x n dict, where n = len(nodes)
  '''
  weights_neighbours = compute_weights_using_number_of_neighbours(nodes, edges)
  weights_features = compute_weights_using_number_of_common_features(nodes, features, norm)
  
  weights = {}
  for u in weights_features.keys():
    weights_for_node = {}
    for z, value in weights_features[u].items():
      weights_for_node[z] = value / weights_neighbours[z]
    weights[u] = weights_for_node

  return weights


# TC_jn
def compute_weights_using_jaccard_similarity_for_neighbours(nodes: List[int], 
                                                            edges: List[Tuple[int, int]]) -> Dict[int, Dict[int, float]]:
  '''number_of_common_neighbours/total_number_of_distinct_neighbours
  
  Output:
    - n x m dict, where n = len(nodes) and m = number_of_neighbours(dict[n])
  '''
  weights = {}

  for u in nodes:
    u_neighbours = [node for node in nodes \
                             if (node, u) in edges or (u, node) in edges]
    weights_for_node = {}
    for z in u_neighbours:
      z_neighbours = [node for node in nodes \
                                if (node, z) in edges or (z, node) in edges]
      intersection = len([node for node in nodes \
                          if node in u_neighbours and node in z_neighbours])
      union = len([node for node in nodes \
                   if node in u_neighbours or node in z_neighbours])
      weights_for_node[z] = intersection / union
    weights[u] = weights_for_node

  return weights


# TC_jf, TC_jfp
def compute_weights_using_jaccard_similarity_for_features(nodes: List[int], 
                                                          features: list) -> Dict[int, Dict[int, float]]:
  '''number_of_common_neighbours/total_number_of_distinct_neighbours
  
  Output:
    - n x n dict
  '''
  weights = {}

  for u in nodes:
    weights_for_node = {}
    for z in nodes:
      if u == z:
        continue
      intersection = len([feature for feature in features[u].keys() \
                          if features[u][feature] & features[z][feature]])
      union = len([feature for feature in features[u].keys() \
                   if features[u][feature] | features[z][feature]])
      weights_for_node[z] = 0 if union is 0 else intersection / union
    weights[u] = weights_for_node

  return weights
