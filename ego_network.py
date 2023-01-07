import os
from typing import Dict, List, Tuple
from text_preprocessing.twitter_text_preprocessing import preprocess_twitter_data


class EgoNetwork: 
  node_id: int
  nodes: List[int]
  circles: List[List[int]]
  edges: List[Tuple[int, int]]
  features: Dict[int, Dict[str, int]]
  preprocessed_features: Dict[int, Dict[str, int]]


def _read_circles(file_path: str) -> List[List[int]]:
  with open(file_path) as file:
    lines = file.readlines()
    return [[int(id) for id in line.strip().split()[1:]] for line in lines]


def _read_edges(file_path: str, node_id: str) -> List[Tuple[int, int]]:
  edges = []
  with open(file_path) as file:
    lines = file.readlines()
    edges = [tuple([int(id) for id in line.strip().split()]) for line in lines]
  # Add an edge from node_id to all the other nodes.
  nodes = sorted(set([node for edge in edges for node in edge]))
  return edges + [(node_id, node) for node in nodes]


def _read_features(ego_features_file_path: str, 
                   features_file_path: str, 
                   feature_names_file_path: str, 
                   node_id: str) -> Dict[int, Dict[str, int]]:
  feature_names = list
  with open (feature_names_file_path) as file:
    lines = file.readlines()
    feature_names = [line.strip().split()[1] for line in lines]

  features = dict()
  with open (ego_features_file_path) as file:
    lines = file.readlines()
    fields = lines[0].strip().split()
    features_for_id = dict()
    for index in range(len(feature_names)):
      features_for_id[feature_names[index]] = int(fields[index])
    features[node_id] = features_for_id

  with open (features_file_path) as file:
    lines = file.readlines()
    for line in lines:
      fields = line.strip().split()
      features_for_id = dict()
      for index in range(len(feature_names)):
        features_for_id[feature_names[index]] = int(fields[index + 1])
      features[int(fields[0])] = features_for_id

  return features


def read_ego_network(node_id: int, dataset_path: str, dataset='twitter') -> EgoNetwork:
  circles_file_path = os.path.join(dataset_path, 
    (str(node_id) + '.circles'))
  edges_file_path = os.path.join(dataset_path, 
    (str(node_id) + '.edges'))
  ego_features_file_path = os.path.join(dataset_path, 
    (str(node_id) + '.egofeat'))
  features_file_path = os.path.join(dataset_path, 
    (str(node_id) + '.feat'))
  feature_names_file_path = os.path.join(dataset_path, 
    (str(node_id) + '.featnames'))

  ego_network = EgoNetwork()
  ego_network.circles = _read_circles(circles_file_path)
  ego_network.edges = _read_edges(edges_file_path, node_id)
  ego_network.nodes = sorted(set([node for edge in ego_network.edges for node in edge]))
  ego_network.features = _read_features(ego_features_file_path, 
                                        features_file_path, 
                                        feature_names_file_path,
                                        node_id)

  if dataset == 'twitter':
    raw_feature_names = ego_network.features[node_id].keys()
    feature_names_mapping = \
      dict([(raw_feature_name, preprocess_twitter_data([raw_feature_name])[0]) \
        for raw_feature_name in raw_feature_names])
    ego_network.preprocessed_features = dict()
    for node, values in ego_network.features.items():
      ego_network.preprocessed_features[node] = dict()
      for raw_feature_name, value in values.items():
        ego_network.preprocessed_features[node][feature_names_mapping[raw_feature_name]] = value

  return ego_network
  

def get_ego_ids(dataset_path: str, order_by_edges_size: bool = True) -> List[int]:
  filenames = os.listdir(dataset_path)
  ids_and_sizes = list()
  for filename in filenames:
    if order_by_edges_size and 'edges' not in filename:
      continue

    id = filename.strip().split('.')[0]
    file_size = os.stat(os.path.join(dataset_path, filename)).st_size
    ids_and_sizes.append((int(id), file_size))
  
  if order_by_edges_size:
    ids_and_sizes = sorted(ids_and_sizes, key=lambda x: x[1])
  else:
    ids_and_sizes = sorted(ids_and_sizes, key=lambda x: x[0])

  return [id for (id, _) in ids_and_sizes]
