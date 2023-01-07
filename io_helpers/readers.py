
import numpy


def read_ndarray_from_file(file_path: str) -> numpy.ndarray:
  with open(file_path, 'r') as file:
   return numpy.loadtxt(file)