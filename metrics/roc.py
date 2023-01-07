"""Module based on 
https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html."""
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from ego_network import EgoNetwork
import numpy as np


def plot_empty_roc_curve(ego_network_id=None):
  plot_roc_curve({}, ego_network_id)  


def plot_roc_curve(data, ego_network_id=None):
  '''Plots ROC curve.

  Input:
    - data: dict. data[label]['fpr'], data[label]['tpr']
  '''

  plt.figure(figsize=(15,15))
  lw = 2 

  for (result_label, current_data) in data.items():
    roc_auc = auc(current_data['fpr'], current_data['tpr'])
    label = f"ROC curve {result_label} (area = {roc_auc:.3})"
    plt.plot(
        current_data['fpr'],
        current_data['tpr'],
        lw=lw,
        label=label,
    )

  plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.0])
  plt.xticks(np.linspace(0, 1, 11), fontsize=18)
  plt.yticks(np.linspace(0, 1, 11), fontsize=18)
  plt.xlabel('False Positive Rate', fontsize=18)
  plt.ylabel('True Positive Rate',  fontsize=18)
  if ego_network_id:
    plt.title(f'Receiver Operating Characteristic for {ego_network_id}', fontsize=22)
  else:
    plt.title('Receiver Operating Characteristic', fontsize=22)
  if len(data):
    plt.legend(loc="lower right", prop={'size': 18})
  plt.show()


def compute_fpr_and_tpr(result_set, ego_network: EgoNetwork):
  truths = []
  predictions = []
  for result in result_set:
    first_node = result[0]
    second_node = result[1]
    first_node_circle = [circle for circle in ego_network.circles if first_node in circle]
    if len(first_node_circle):
      first_node_circle = first_node_circle[0]
    truths.append(second_node in first_node_circle)
    predictions.append(result[2])

  if 1 not in truths:
    return (None, None)
  fpr, tpr, _ = roc_curve(truths, predictions, drop_intermediate=False)
  return (fpr, tpr)


def plot_roc_curve_for_triadic_closeness(triadic_closeness_results: dict, 
                                         ego_network: EgoNetwork):
    data = {}

    for result_label, result_set in triadic_closeness_results.items():
      fpr, tpr = compute_fpr_and_tpr(result_set, ego_network)
      if fpr is None:
        continue
      data[result_label] = {'fpr': fpr, 'tpr': tpr}
      data[result_label]['fpr'] = fpr
      data[result_label]['tpr'] = tpr
    
    plot_roc_curve(data)
