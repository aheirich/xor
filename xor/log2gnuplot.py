#
# log2gnuplot.py
#

import sys
import numpy
from scipy.spatial import ConvexHull

l0 = 2



def distance(p0, p1):
  d = 0
  for i in range(len(p0)):
    x = p0[i]
    y = p1[i]
    diff = x - y
    d = d + diff * diff
  return d


def closestPointIndex(points, point):
  closestPointIndex = 0
  if point is None:
    return closestPointIndex
  minDistance = 999999
  
  for i in range(len(points)):
    if points[i] is None:
      continue
    d = distance(point, points[i])
    if d < minDistance:
      minDistance = d
      closestPointIndex = i

  return closestPointIndex


def sortedPoints(points):
  result = []
  point = None
  for i in range(len(points)):
    j = closestPointIndex(points, point)
    point = points[j]
    points[j] = None
    result.append(point)
  result.append(result[0])
  return result


def convexHull(points):
  vertices = ConvexHull(numpy.array(points)).vertices
  result = []
  for v in vertices:
    result.append(points[v])
  result.append(result[0])
  return result

def outputPoints(points):
  
  cHull = convexHull(points)
  for k in range(len(cHull)):
    p = cHull[k]
    string = ""
    for x in p:
      string = string + str(x) + " "
    print string
  print ""


points = []
lines = sys.stdin.readlines()

for k in range(len(lines)):
  line = lines[k]
  
  if line.startswith("'receptive field"):
    if len(points) > 0:
      outputPoints(points)
    points = []

  if line.startswith("z0 [*] :="):
    point = []
    for i in range(l0):
      line = lines[k + 1]
      k = k + 1
      words = line.split(' ')
      x = float(words[2].strip())
      point.append(x)
    points.append(point)

