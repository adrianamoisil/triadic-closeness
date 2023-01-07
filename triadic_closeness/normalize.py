from sklearn.preprocessing import normalize


def normalize_triadic_closeness(results: list, norm: str) -> list:
  all_triadic_closeness = [result[2] for result in results] 
  all_triadic_closeness_normalized = normalize([all_triadic_closeness],norm=norm)[0]
  normalized_results = []
  for i in range(len(results)):
    normalized_result = (results[i][0], results[i][1], all_triadic_closeness_normalized[i])
    normalized_results.append(normalized_result)
  return normalized_results
  
 
def normalize_triadic_closeness_using_l1(results: list):
  return normalize_triadic_closeness(results, 'l1')
  
 
def normalize_triadic_closeness_using_l2(results: list):
  return normalize_triadic_closeness(results, 'l2')
  
 
def normalize_triadic_closeness_using_max(results: list):
  return normalize_triadic_closeness(results, 'max')
