from typing import Optional
from triadic_closeness.triadic_closeness import compute_triadic_closeness_for_all_nodes


class TriadicClosenessClassifier:
  def __init__(self, edges: list, should_normalize: bool = True):
    self._edges = edges
    self._should_normalize = should_normalize
    self._triadic_closeness = compute_triadic_closeness_for_all_nodes(
        self._edges, self._should_normalize
    )


  def get_triadic_closeness(self):
    return self._triadic_closeness


  def predict(self, first_node, second_node) -> Optional[float]:
    triadic_closeness = [triadic_closeness for triadic_closeness in self._triadic_closeness \
      if triadic_closeness[0] == first_node and triadic_closeness[1] == second_node]
    if len(triadic_closeness):
      return triadic_closeness[0][2]
    return None
