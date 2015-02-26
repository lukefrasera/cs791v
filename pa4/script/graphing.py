#!/usr/bin/env python
import csv, sys
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

class ExtractError(Exception):
  def __init__(self, value):
    self.value = value
  def __str__(self):
    return repr(self.value)

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
  N_indexs = [(0, data[0])]
  for i, number in enumerate(data):
    if number != prev_number:
      N_indexs.append((i,number))
    prev_number = number
  N_indexs.append((len(data), -1))
  return N_indexs

def ExtractDataRange(n_index, v_index, t_index, N, version, thresh):
  n_range = (-1, -1)
  for i, (index, value) in enumerate(n_index):
    if value == N:
      n_range = (index, n_index[i+1][0])
      break
  if n_range == (-1, -1):
    raise ExtractError("Error: N index Doesn't exist")
  v_range = (-1, -1)
  for i, (index, value) in enumerate(v_index):
    if value == version and index >= n_range[0] and index <= n_range[1]:
      v_range = (index, v_index[i+1][0])
      break
  if v_range == (-1, -1):
    raise ExtractError("Error: Version index Doesn't exist")
  t_range = (-1, -1)
  for i, (index, value) in enumerate(t_index):
    if value == thresh and index >= v_range[0] and index <= v_range[1]:
      t_range = (index, t_index[i+1][0])
      break
  if t_range == (-1, -1):
    raise ExtractError("Error: Version index Doesn't exist")
  return t_range

def main(argv):
  try:
    raw_data, header = ImportCSVFile(argv[1])
  except IndexError:
    print "Error: You must provide a 'csv' file to load"
    sys.exit(0)
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
  index_range = ExtractDataRange(N_index, V_index, T_index, 1048576, 1, 1)

  
  sub_threads = []
  sub_blocks = []
  sub_samples = []
  row = []
  brow = []
  srow = []
  for i in xrange(index_range[0], index_range[1]):
    if i > len(threads)-2:
      row.append(threads[i])
      brow.append(blocks[i])
      srow.append(samples[i])

      sub_threads.append(row)
      sub_blocks.append(brow)
      sub_samples.append(srow)

      row = []
      brow = []
      srow = []
    elif threads[i] < threads[i+1]:
      row.append(threads[i])
      brow.append(blocks[i])
      srow.append(samples[i])
    else:
      row.append(threads[i])
      brow.append(blocks[i])
      srow.append(samples[i])

      sub_threads.append(row)
      sub_blocks.append(brow)
      sub_samples.append(srow)

      row = []
      brow = []
      srow = []

  # graph data
  X = np.array(sub_threads)
  Y = np.array(sub_blocks)
  # Z = np.array(np.log1p(sub_samples))
  Z = np.array(sub_samples)
  print Z

  fig = plt.figure()
  ax = fig.gca(projection='3d')
  surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
          linewidth=0, antialiased=True)
  # ax.set_zlim(-1.01, 1.01)
  ax.set_xlabel('Threads')
  ax.set_xlim(1, 1024)
  ax.set_ylabel('Blocks')
  ax.set_ylim(1, 32768)
  ax.set_zlabel('Execution Time(ms)')
  # ax.zaxis.set_major_locator(LinearLocator(10))
  # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

  fig.colorbar(surf, shrink=0.5, aspect=10)

  plt.show()


if __name__ == '__main__':
  main(sys.argv)