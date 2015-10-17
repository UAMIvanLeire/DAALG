import numpy as np
import Queue as pq
import timeit as timeit
import math
import fit
import matplotlib.pyplot as plt
from sklearn import linear_model

#sparseFactor = numero entre 0 y 1 np.random.binomial(1, sparseFActor) ==1

def randMatrPosWGraph(nNodes, sparseFactor, maxWeight=50.):
  
  matriz = np.zeros(shape = (nNodes, nNodes))
  for i in range(nNodes):
    for j in range(nNodes):
      matriz[i][j] = np.inf
  for i in range(nNodes):
    for j in range(nNodes):
      if i != j:  
        if np.random.binomial(1, sparseFactor) == 1:
          weight  = np.random.randint(0, maxWeight)
          neight = np.random.randint(0, nNodes)
          if i != neight:
            matriz[i][neight] = weight
  return matriz



def cuentaRamas(mG):
  count = 0
  for i in range(len(mG)):
    for j in range(len(mG)):
      if mG[i][j] != np.inf:
        count = count + 1
  return count

def fromAdjM2Dict(mG) :
  dic = {}
  
  for i in range(len(mG)):
    dicAux = {}
    for j in range(len(mG)):
      if mG[i][j] != np.inf:
        dicAux.update({j:mG[i][j]}) 
    dic.update({i:dicAux})
  return dic

def fromDict2AdjM(dG):
  
  nNodes = len(dG.keys())
  matriz = np.zeros(shape = (nNodes, nNodes))
  for i in range(nNodes):
    for j in range(nNodes):
      matriz[i][j] = np.inf

  for i in range(nNodes):
      dicAux = dG[i]
      for k in dicAux:
        if dicAux[k] != {}:
          matriz[i][k] = dicAux[k]
  return matriz

def dijkstraM(mG, u):
  v = []
  p = []
  d = []
  for i in range(len(mG)):
    v.append(False)
    p.append(None) 
    d.append(np.inf) 

  q = pq.PriorityQueue()
 
  d[u] = 0
  

  q.put((mG[u][0], u)) 

  while not q.empty():
    w = q.get()
    if v[w[1]] == False:
      v[w[1]] = True
      for k in range(len(mG[w[1]])):
        if d[k] > d[w[1]] + mG[w[1]][k]:
          d[k] = d[w[1]] + mG[w[1]][k]
          p[k] = w[1]
          q.put((mG[k][0], k))


  return p, d

def dijkstraD(dG, u):

  matriz = fromDict2AdjM(dG)
  return dijkstraM(matriz)
 

arr = []
k = 0

def timeDijkstraM(nGraphs, nNodesIni, nNodesFin, step, sparseFactor=.25): 
  timeList = []

  for nNodes in range(nNodesIni, nNodesFin, step):
    matriz = randMatrPosWGraph(nNodes, sparseFactor, maxWeight=50.)
    arr.append(matriz)
    # print arr
    acum = 0
    for j in range(len(matriz)):
      k = j
      setup = "import __main__ as test"
      
      acum += timeit.timeit("test.dijkstraM(test.arr[len(test.arr)-1], test.k)", setup=setup, number = 1)

    timeList.append(acum/len(matriz))
        
          

  return timeList
  
def n2log(nNodesIni, nNodesFin, step):
  log = []
  for i in range(nNodesIni, nNodesFin, step):
    log.append(i**2 * np.log10(i))
  nlog = np.array(log);
  return nlog

def plotFitDijkstraM(l, func2fit, nNodesIni, nNodesFin, step):
  n2log = func2fit(nNodesIni, nNodesFin, step);
  print "medidas"
  print l 
  print len(l)
  print n2log
  print len(n2log)
  N = len(l) 

  lr = linear_model.LinearRegression()
  lr.fit(n2log[:, np.newaxis], l)
  fit = lr.predict(n2log[:, np.newaxis])

  plt.plot(fit)
  plt.plot(l)
  plt.show()

  return


def dijkstraMAllPairs(mG):
  nNodes = len(mG)
  # print mG
  matriz = np.zeros(shape = (nNodes, nNodes))
  for i in range(nNodes):
    for j in range(nNodes):
      matriz[i][j] = np.inf

  for i in range(len(mG)):
    p,d = dijkstraM(mG,i)
    for j in range(len(mG)):
      matriz[i][j] = d[j]
  return matriz

# FIX
def floydWarshall(mG):
  nNodes = len(mG)
  # print mG
  matriz = np.zeros(shape = (nNodes, nNodes))
  for i in range(nNodes):
    for j in range(nNodes):
      matriz[i][j] = np.inf

  for k in range(nNodes):
    matriz[k][k] = 0
  for i in range(nNodes):
    for j in range(nNodes):
      if i != j:
        matriz[i][j] = mG[i][j]


  for k in range(nNodes):
    for i in range(nNodes):
      for j  in range(nNodes):
        aux = matriz[i][k] + matriz[k][j]
        if matriz[i][j] > aux:
          matriz[i][j] = aux
  
  return matriz


# FIX mirar donde usar nGraphs
def timeDijkstraMAllPairs(nGraphs, nNodesIni, nNodesFin, step, sparseFactor):
  timeList = []

  for nNodes in range(nNodesIni, nNodesFin, step):
    matriz = randMatrPosWGraph(nNodes, sparseFactor, maxWeight=50.)
    arr.append(matriz)
    # print arr
    acum = 0
    for j in range(len(matriz)):
      k = j
      setup = "import __main__ as test"
      
      acum += timeit.timeit("test.dijkstraMAllPairs(test.arr[len(test.arr)-1])", setup=setup, number = 1)

    timeList.append(acum/len(matriz))
  return timeList




def timeFloydWarshall(nGraphs, nNodesIni, nNodesFin, step, sparseFactor):
  timeList = []

  for nNodes in range(nNodesIni, nNodesFin, step):
    matriz = randMatrPosWGraph(nNodes, sparseFactor, maxWeight=50.)
    arr.append(matriz)
    # print arr
    acum = 0
    for j in range(len(matriz)):
      k = j
      setup = "import __main__ as test"
      
      acum += timeit.timeit("test.floydWarshall(test.arr[len(test.arr)-1])", setup=setup, number = 1)

    timeList.append(acum/len(matriz))
  return timeList


def dG2TGF(dG, fName):
  f = open(fName, "w")
  matriz = fromDict2AdjM(dG)
  for i in range(len(matriz)):
    f.write("%d" % i)
    f.write("\n")
  f.write("#")
  f.write("\n")

  for i in range(len(matriz)):
    for j in range(len(matriz)):
      if matriz[i][j] != np.inf:
        f.write("%d" % i)
        f.write(" ")
        f.write("%d" % j)
        f.write(" ")
        f.write("%s" % matriz[i][j])
        f.write("\n")
  f.close()
  return

def TGF2dG(fName):
  f = open(fName)
  initialize = 0
  nNodes = 0
  for line in f.read().splitlines():
    spl = line.split()
    if len(spl) == 1:
      nNodes = 1 + nNodes
    else:
      initialize = initialize + 1
    if initialize == 1:
      nNodes = nNodes - 1
      matriz = np.zeros(shape = (nNodes, nNodes))

      for i in range(nNodes):
        for j in range(nNodes):
          matriz[i][j] = np.inf
    if initialize > 0:
      matriz[int(spl[0])][int(spl[1])] = spl[2]

  return matriz


mG = randMatrPosWGraph(4, 0.5, maxWeight= 50)

m = np.zeros(shape = (4, 4))
for i in range (4): m[i] = np.inf

m[0][1] = 2
m[1][0] = 2
m[1][2] = 4
m[2][1] = 6
m[2][3] = 3
m[3][1] = 2
mG = m
print m 



# print cuentaRamas(mG)
dG = fromAdjM2Dict(m)
# print fromDict2AdjM(fromAdjM2Dict(mG))
# print dijkstraM(m,3)
# print "timeDijkstraM"
# time = timeDijkstraM(4, 1, 100, 10)
# print fromAdjM2Dict(m)
# print "dijkstraMAllPairs"
# print dijkstraMAllPairs(m)
# print "floydWarshall"
# print floydWarshall(m)
# plotFitDijkstraM(time, n2log, 1, 100, 10)
# print "timeDijkstraMAllPairs"
# timeAllPairs = timeDijkstraMAllPairs(4, 1, 100, 10, .25)
# print timeAllPairs
# plotFitDijkstraM(timeAllPairs, n2log, 1, 100, 10)
# print "timeFloydWarshall"
# timeFloyd = timeFloydWarshall(4, 1, 100, 10, .25)
# print timeFloyd
# plotFitDijkstraM(timeFloyd, n2log, 1, 100, 10)

dG2TGF(dG, "prueba.txt")
print TGF2dG("prueba.txt")
# timeDijkstraMAllPairs
# [0.00011014938354492188, 0.0015827092257412996, 0.009857120968046643, 0.027381089425856067, 0.05817905867972025, 0.10889414712494495, 0.16937813993360176, 0.2580439204901037, 0.3765948878394233, 0.5240651277395395]
# timeFloydWarshall
# [7.414817810058594e-05, 0.0015359358354048295, 0.009419327690487816, 0.0272390688619306, 0.059274597865779224, 0.10464866021100212, 0.17436315192550908, 0.25969763540885815, 0.37058487350558056, 0.525374289397355]