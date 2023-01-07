
import numpy


def write_ndarray_to_file(data: numpy.ndarray, file_path: str):
  with open(file_path, 'w') as file:
    numpy.savetxt(file, data)