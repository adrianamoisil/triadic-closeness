# Triadic Closeness
This is a python library that implements an altered version of the Triadic Closeness algorithm on a Social Network that can be seen as a directed graph in which we have some extra information about each node, besides its connections.


# Jupyter Notebooks
There are three jupyter notebooks:
- *analyze_data.ipynb*
- *visualize_roc.ipynb*
- *generate_statistics.ipynb*

Before running these notebook, make sure that the path where the dataset is located and the one where results should be generated are correct.

# Dataset
The original dataset used is the Ego Twitter dataset that can be found at https://snap.stanford.edu/data/ego-Twitter.html. Certain snippets of code are written specifically for this dataset (for example *ego_network.py*). However, one can easily adapt the existing logic to make it work with a different dataset.

