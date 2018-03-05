#
# random_start.py
#
# generate random initial conditions for XOR inversion
#

import random

l1 = 16 # number of hidden units
l0 = 2 # num input units
numPointsPerSegment = 5
numReceptiveFieldSamples = 4
modelFile = "../xor_ampl.mod"
dataFile = "../xor_ampl.dat"
solver = "/Users/aheirich/Desktop/amplide.macosx64/minos"


def generateFile(filename, target):
  f = open(filename, "w")
  f.write("# XOR relu network initialization\n")
  f.write("\n")
  
  f.write("param y_target := \n")
  for i in range(len(target)):
    f.write(str(i + 1) + " " + str(target[i]) + "\n")
  f.write(";\n")

  f.write("\n")
  f.write("var z2 := \n")
  for i in range(len(target)):
    f.write(str(i + 1) + " " + str(target[i]) + "\n")
  f.write(";\n")
  
  f.write("\n")
  f.write("var z1 := \n")
  for i in range(l1):
    randInt = random.randint(0, 9999)
    randomValue = randInt / 5000.0 - 1.0
    f.write(str(i + 1) + " " + str(randomValue) + "\n")
  f.write(";\n")

  f.write("\n")
  f.write("var z0 := \n")
  for i in range(l0):
    randInt = random.randint(0, 9999)
    randomValue = randInt / 5000.0 - 1.0
    f.write(str(i + 1) + " " + str(randomValue) + "\n")
  f.write(";\n")

  f.close()


def generateFiles(filenameBase, points, script):
  script.write("# points " + str(points) + "\n")
  caseLabels = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
  for j in range(numReceptiveFieldSamples):
    script.write("display \"receptive field " + filenameBase + " sample " + str(j) + "\" ;\n")
    for i in range(len(points)):
      script.write("# " + filenameBase + " receptive field sample " + str(j) + " point " + str(i) + "\n")
      script.write("reset ;\n")
      script.write("option solver \"" + solver + "\" ;\n")
      script.write("model \"" + modelFile + "\" ;\n")
      script.write("data \"" + dataFile + "\" ;\n")
      filename = filenameBase + str(j) + caseLabels[i] + ".dat"
      generateFile(filename, points[i])
      script.write("data \"" + filename + "\" ;\n")
      script.write("solve ;\n")
      script.write("display z0 ;\n")

def connectPoints(base, point, result):
  increment = []
  for i in range(len(base)):
    x = float(base[i])
    y = float(point[i])
    increment.append((y - x) / numPointsPerSegment)
  for k in range(numPointsPerSegment):
    newPoint = []
    for j in range(len(base)):
      x = base[j]
      i = increment[j]
      newPoint.append(x + i * k)
    result.append(newPoint)
  return result


def closedContour(corners):
  result = []
  for i in range(len(corners)):
    if i == 0:
      continue
    previous = corners[i - 1]
    point = corners[i]
    result = connectPoints(previous, point, result)
  result = connectPoints(point, corners[0], result)
  return result


# generate ampl script to find all point inversions
amplScript = open("ampl_compute_XOR_receptive_fields.run", "w")
amplScript.write("# ampl script to compute XOR receptive fields\n")

# positive XOR
XOR_true = closedContour([ [.6, .1], [.6, .4], [.9, .4], [.9, .1] ])
XOR_false = closedContour([ [.1, .6], [.1, .9], [.4, .9], [.4, .6] ])

generateFiles("XOR_true_", XOR_true, amplScript)
generateFiles("XOR_false_", XOR_false, amplScript)

amplScript.close()


