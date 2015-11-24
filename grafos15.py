#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import Queue as pq
import timeit as timeit
import math
import matplotlib.pyplot as plt
#from sklearn import linear_model
import argparse 



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
  dic = dict()
  contador = 0
  
  for i in mG: #range(mG.shape[0]):
    dicAux =[]
    #dic.update( {i:[]} )
    for j in range(len(i)): #range(N):
      #if j != i and mG[i,j] != np.inf:
      #  dic[i].append( (j, mG[i,j]) )
      dicAux2 = []

      if contador != j and i[j] != np.inf:
        dicAux2.append(i[j])
        dicAux2.append(j)
        dicAux.append((i[j], j))

    dic.update({contador : dicAux})
    contador += 1
  return dic

def fromDict2AdjM(dG):
  
  nNodes = len(dG.keys())
  matriz = np.zeros(shape = (nNodes, nNodes))
  for i in range(nNodes):
    matriz[i] = np.inf

  #for k in dG:
  # for le in dG[k]:
  #   j = le[0]; w = le[1]
  #   matriz[k, j] = w
  for key, value in dG.iteritems():
    while value:
      lista = value.pop()
      #lista = list(lista)
    
      matriz[key][lista[0]] = lista[1]
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
  return dijkstraM(matriz, u)
 

arr = []
k = 0

def timeDijkstraM(nGraphs, nNodesIni, nNodesFin, step, sparseFactor=.25): 
  timeList = []

  for nNodes in range(nNodesIni, nNodesFin, step):
    matriz = randMatrPosWGraph(nNodes, sparseFactor, maxWeight=50.)
    arr.append(matriz)
    
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

def plotFitDijkstraM(lT, func2fit, nNodesIni, nNodesFin, step):
  n2log = func2fit(nNodesIni, nNodesFin, step);
  print "medidas"
  print l 
  print len(l)
  print n2log
  print len(n2log)
  N = len(l) 

  lr = linear_model.LinearRegression()
  lr.fit(n2log[:, np.newaxis], lT)
  fit = lr.predict(n2log[:, np.newaxis])

  plt.plot(fit)
  plt.plot(lT)
  plt.show()

  return


def dijkstraMAllPairs(mG):
  nNodes = len(mG)
  matriz = np.zeros(shape = (nNodes, nNodes))
  for i in range(nNodes):
    for j in range(nNodes):
      matriz[i][j] = np.inf

  for i in range(len(mG)):
    p,d = dijkstraM(mG,i)
    for j in range(len(mG)):
      matriz[i][j] = d[j]
  return matriz


def floydWarshall(mG):
  nNodes = len(mG)
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

  return fromDict2AdjM(matriz)



#plotFitDijkstraM(timeDijkstraM(20, 10, 1000, 10, sparseFactor=.25), n2log, 10, 1000, 10)


# parser = argparse.ArgumentParser(description='Parseo de argumentos')
# parser.add_argument('file')

# args = parser.parse_args()

# mG = TGF2dG(args.file)
# dicc = fromAdjM2Dict(mG)
# print mG
# print fromDict2AdjM(dicc)

# p = dijkstraMAllPairs(mG)
# r = floydWarshall(mG)

# print p[0:8, 0:8]
# print r[0:8, 0:8]

def randMatrUndPosWGraph(nNodes, sparseFactor, maxWeight=50.):

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
            matriz[neight][i] = weight
  return matriz


def checkUndirectedM(mG):
  for i in range(len(mG)):
    for j in range(len(mG)):
      if mG[i][j] != mG[j][i]:
        return False
  return True

def checkUndirectedD(dG):   
  for k in dG:
    for le in dG[k]:
      nodo = le[1]
      peso = le[0]
      count = 0
      for le2 in dG[nodo]:  
        peso2 = le2[0]
        nodo2 = le2[1]
        if k == nodo2:
          if peso == peso2:
            count = 1
      if count == 0:
        return False
  return True

def initCD(N):
  array = np.ones(N)*-1
  return array

def union(rep1, rep2, pS):
  print "Dentro de union"
  if pS[rep2] < pS[rep1]: #T_rep2 is taller
    print "11"
    pS[rep1] = rep2
    return rep2
  elif pS[rep2] > pS[rep1]: #T_rep1 is taller
    print "22"
    pS[rep2] = rep1
    return rep1
  else: #T_rep1, T_rep2 have the same lenght
    pS[rep2] = rep1
    pS[rep1] -= 1
    print "33"
    return rep1
  
def find(ind, pS, flagCC):
  if flagCC == False:
    while pS[ind] != -1:
      print ind
      ind = pS[ind]
    return ind
  else:
    z = ind
    while pS[z] >= 0:
      z = pS[z]
    while pS[ind] >= 0:
      y = pS[ind]
      pS[ind] = z
      ind = y
    return z

#KRUSKAL

def insertPQ(dG, Q):
  #Recorrer el arbol hallando uniones
  #Q.put((mG[u][0], u)) 
  for k in dG:
    for le in dG[k]:
      nodo = le[1]
      peso = le[0]
      if(k < nodo):
        Q.put((peso,k,nodo)) 
  return 

def kruskal(dG, flagCC=True):
  L = [] #we save the MST in L
  Q = pq.PriorityQueue()  #iniPQ(Q)
  insertPQ(dG, Q)
  S = initCD(len(dG.keys())) #initDS(V, S) # 2
  print S
  # while not Q.empty():
  #   print Q.get()
  print Q.qsize()
  while not Q.empty(): # 3
    w = Q.get() #extPQ((u,v), Q) # 4
    print w
    print Q.qsize()    
    x = find(w[1], S, flagCC)
    y = find(w[2], S, flagCC) # 5
    print "xy"
    print x
    print y
    if x != y:
      L.append((w[1],w[2])) #L.append( (u, v) ) # 6
      #union(x, y, S) # 7
  return L


#TEST

m = np.zeros(shape = (4, 4))
for i in range (4): m[i] = np.inf

m[0][1] = 2
m[1][0] = 3
m[1][2] = 1
m[2][1] = 4
m[2][3] = 2
m[3][1] = 1

print m 

dicc = fromAdjM2Dict(m)
print dicc

checkUndirectedD(dicc)

#Prueba diccionarios no dirigidos

mG = randMatrUndPosWGraph(4, 0.5, maxWeight= 50)

udicc = fromAdjM2Dict(mG)
print udicc
print checkUndirectedM(mG)
print checkUndirectedD(udicc)
print checkUndirectedD(dicc)

#Prueba CD
#FIX-IT representación CD no esta claro, comprobacion con Kruskal
pS = [[1, 2 ,3], [8, 4, 5], [6, 0, 9]]
print initCD(5)
print union(0, 1, pS)
#print find(1, pS, False)


#Test Kruskal
# Q = pq.PriorityQueue()
# insertPQ(udicc, Q)
# while not Q.empty():
#   print Q.get()
udicc2 = {0:[(10,1), (12,2), (5,1)], 1:[(10,0),(4,3),(5,0)], 2:[(12,0)], 3:[(4,1)]}
print checkUndirectedD(udicc2)
print "LLLLL"
print kruskal(udicc2, flagCC=False)















