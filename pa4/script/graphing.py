#!/usr/bin/env python
import csv, sys
import matplotlib.pyplot as pyplot
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def SurfaceDataManip(row, col, values):
  pass
def ImportCSVFile(filename):
  result = []
  header = []
  with open(filename) as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for i,row in enumerate(reader):
      if i > 0:
        try:
          result.append([int(element) if i is not 5 else float(element) for i, element in enumerate(row)])
        except ValueError:
          continue
      else:
        header = row
  return result, header

def IndexChangeList(data):
  prev_number = data[0]
  N_indexs = []
  for i, number in enumerate(data):
    if number != prev_number:
      N_indexs.append(i)
    prev_number = number
  return N_indexs
def main(argv):
  raw_data, header = ImportCSVFile(argv[1])
  data = np.array(raw_data) 
  N = data[:,0].astype(int)
  version = data[:,1].astype(int)
  threshold = data[:,2].astype(int)
  blocks = data[:,3].astype(int)
  threads = data[:,4].astype(int)
  samples = data[:,5]

  N_index = IndexChangeList(N)
  V_index = IndexChangeList(version)
  T_index = IndexChangeList(threshold)
  print N_index
  print V_index
  print T_index


if __name__ == '__main__':
  main(sys.argv)