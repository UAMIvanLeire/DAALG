#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import Queue as pq
import timeit as timeit
import math
import matplotlib.pyplot as plt
#from sklearn import linear_model
import argparse 


###############################################################################
# Nombre: randMatrPosWGraph
# Funcionalidad: Genera la matriz de adyacencia de un grafo no dirigido.
# Argumentos: 
#             -nNodes: Número de nodos del grafo a generar.
#             -sparseFactor: factor que determinará la densidad del grafo.
# Salida: 
#             -matriz: matriz de adyacencia del grafo generado
###############################################################################
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

###############################################################################
# Nombre: cuentaRamas
# Funcionalidad: Genera la matriz de adyacencia de un grafo no dirigido.
# Argumentos: 
#             -mG: matriz de adyacencia del grafo a tratar.
# Salida: 
#             -count: número de ramas(excluyendo diagonales) del grafo. 
###############################################################################
def cuentaRamas(mG):
  count = 0
  for i in range(len(mG)):
    for j in range(len(mG)):
      if mG[i][j] != np.inf:
        count = count + 1
  return count

###############################################################################
# Nombre: fromAdjM2Dict
# Funcionalidad: Genera un diccionario de adyacencia de un grafo a partir de
# la matriz de adyacencia de ese mismo grafo.
# Argumentos: 
#             -mG: matriz de adyacencia del grafo a tratar.
# Salida: 
#             -dic: diccionario de adyacencia equivalente a la matriz dada.
###############################################################################
def fromAdjM2Dict(mG) :
  dic = dict()
  contador = 0
  
  for i in mG: 
    dicAux =[]
    for j in range(len(i)): 
      dicAux2 = []

      if contador != j and i[j] != np.inf:
        dicAux2.append(i[j])
        dicAux2.append(j)
        dicAux.append((i[j], j))

    dic.update({contador : dicAux})
    contador += 1
  return dic

##############################################################################
# Nombre: fromDict2AdjM
# Funcionalidad: Genera una matriz de adyacencia de un grafo a partir de
# el diccionario de adyacencia de ese mismo grafo.
# Argumentos: 
#             -dG: diccionario de adyacencia del grafo a tratar.
# Salida: 
#             -matriz: matriz de adyacencia equivalente al diccionario dado.
###############################################################################
def fromDict2AdjM(dG):
  
  nNodes = len(dG)
  matriz = np.zeros(shape = (nNodes, nNodes))

  for i in range(nNodes):
    matriz[i] = np.inf

  for k in dG:
    print k
    for le in dG[k]:
      j = le[0]
      w = le[1]
      matriz[k][w] = j
  
  return matriz

##############################################################################
# Nombre: dijkstraM
# Funcionalidad: Ejecuta el algorimo codicioso Dijkstra sobre el grafo repre-
# sentado por su matriz de adyacencia.
# Argumentos: 
#             -mG: matriz de adyacencia del grafo a tratar.
#             -u: nodo desde el que ejecutar el algoritmo.
# Salida: 
#             -p: Lista de nodos previos del recorrido del algoritmo por el grafo.
#             -d: Lista de distancias mínimas desde el nodo u al resto de nodos.
###############################################################################
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

##############################################################################
# Nombre: dijkstraD
# Funcionalidad: Ejecuta el algorimo codicioso Dijkstra sobre el grafo repre-
# sentado por su diccionario de adyacencia.
# Argumentos: 
#             -dG: diccionario de adyacencia del grafo a tratar.
#             -u: nodo desde el que ejecutar el algoritmo.
# Salida: 
#             -p: Lista de nodos previos del recorrido del algoritmo por el grafo.
#             -d: Lista de distancias mínimas desde el nodo u al resto de nodos.
###############################################################################
def dijkstraD(dG, u):

  matriz = fromDict2AdjM(dG)
  return dijkstraM(matriz, u)

arr = []
k = 0

##############################################################################
# Nombre: timeDijkstraM
# Funcionalidad: Ejecuta el algorimo codicioso Dijkstra sobre NGraphs grafos, 
# con tamaño variable entre nNodesIni y nNodesFin de step en step, con un
# sparseFactor dado, y mide los tiempos de ejecución del algoritmo en cada una
# de sus iteraciones.
# Argumentos: 
#             -nGraphs: número de grafos sobre los operar el algortimo.
#             -nNodesIni: Número de nodos mínimo.
#             -nNodesFin: Número de nodos máximo.
#             -step: Numero de nodos de diferencia entre una iteración y otra.
#             -sparseFactor: factor de dispersión para la generación de grafos.
# Salida: 
#             -timeList: lista de tiempos de ejecución de las iteraciones.
###############################################################################
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

##############################################################################
# Nombre: n2log
# Funcionalidad: Función auxiliar para la generación de datos para la aproxi-
# macion con la función matemática n2log en la función plotFitDijkstraM.
# Argumentos: 
#             -nNodesIni: Número de nodos mínimo.
#             -nNodesFin: Número de nodos máximo.
#             -step: Numero de nodos de diferencia entre una iteración y otra.
#   
# Salida: 
#             -nlog: Lista de los resultados de las iteraciones de la función 
#             n2log.
###############################################################################
def n2log(nNodesIni, nNodesFin, step):
  log = []
  for i in range(nNodesIni, nNodesFin, step):
    log.append(i**2 * np.log10(i))
  nlog = np.array(log);
  return nlog

##############################################################################
# Nombre: plotFitDijkstraM
# Funcionalidad: Realiza una gráfica aproximando los resultados en tiempos 
# obtenidos en lT con la función matemática func2fit.
# Argumentos: 
#             -lT: lista de tiempos de ejecución del algortimo a representar.
#             -func2fit: Función matemática de aproximación para la gráfica.
#             -nNodesIni: Número de nodos mínimo.
#             -nNodesFin: Número de nodos máximo.
#             -step: Numero de nodos de diferencia entre una iteración y otra.
#   
# Salida: 
#             -timeList: lista de tiempos de ejecución de las iteraciones.
###############################################################################
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

##############################################################################
# Nombre: dijkstraMAllPairs
# Funcionalidad: Ejecuta el algorimo codicioso Dijkstra sobre el grafo repre-
# sentado por su diccionario de adyacencia.
# Argumentos: 
#             -mG: Matriz de adyacencia del grafo a recorrer.
# Salida: 
#             -matriz: Matriz de distancias mínimas desde todos los nodos a 
#             todos los nodos.
###############################################################################
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

##############################################################################
# Nombre: floydWarshall
# Funcionalidad: Ejecuta el algorimo Floyd-Warshall sobre el grafo mG.
# Argumentos: 
#             -mG: Matriz de adyacencia del grafo a recorrer.
# Salida: 
#             -matriz: Matriz de distancias mínimas desde todos los nodos a 
#             todos los nodos.
###############################################################################
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


##############################################################################
# Nombre: timeDijkstraMAllPairs
# Funcionalidad: Ejecuta el algorimo codicioso Dijkstra iterado sobre NGraphs 
# grafos, con tamaño variable entre nNodesIni y nNodesFin de step en step, 
# con un sparseFactor dado, y mide los tiempos de ejecución del algoritmo en 
# cada una de sus iteraciones.
# Argumentos: 
#             -nGraphs: número de grafos sobre los operar el algortimo.
#             -nNodesIni: Número de nodos mínimo.
#             -nNodesFin: Número de nodos máximo.
#             -step: Numero de nodos de diferencia entre una iteración y otra.
#             -sparseFactor: factor de dispersión para la generación de grafos.
# Salida: 
#             -timeList: lista de tiempos de ejecución de las iteraciones.
###############################################################################
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



##############################################################################
# Nombre: timeFloydWarshall
# Funcionalidad: Ejecuta el algorimo Floyd-Warshall sobre NGraphs 
# grafos, con tamaño variable entre nNodesIni y nNodesFin de step en step, 
# con un sparseFactor dado, y mide los tiempos de ejecución del algoritmo en 
# cada una de sus iteraciones.
# Argumentos: 
#             -nGraphs: número de grafos sobre los operar el algortimo.
#             -nNodesIni: Número de nodos mínimo.
#             -nNodesFin: Número de nodos máximo.
#             -step: Numero de nodos de diferencia entre una iteración y otra.
#             -sparseFactor: factor de dispersión para la generación de grafos.
# Salida: 
#             -timeList: lista de tiempos de ejecución de las iteraciones.
###############################################################################
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

###############################################################################
# Nombre: dG2TGF
# Funcionalidad: Recibe la lista de adyacencia de un grafo ponderado y guarda
# dicho grafo en un archivo de nombre fName en formato TGF.
# Argumentos: 
#             -fName: Nombre del fichero a editar.
#             -dG: Diccionario de adyacencia del grafo guardar. 
# Salida: 
#             -matriz: matriz de adyacencia del grafo leído.
###############################################################################
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

###############################################################################
# Nombre: TGF2dG
# Funcionalidad: Lee un grafo ponderado guardado en el archivo fName en formato
#  TGF y devuelve la lista de adyacencia del mismo.
# Argumentos: 
#             -fName: Nombre del fichero a leer. 
# Salida: 
#             -matriz: matriz de adyacencia del grafo leído.
###############################################################################
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

  return fromAdjM2Dict(matriz)

###############################################################################
# Nombre: randMatrUndPosWGraph
# Funcionalidad: Genera la matriz de adyacencia de un grafo dirigido.
# Argumentos: 
#             -nNodes: Número de nodos del grafo a generar.
#             -sparseFactor: factor que determinará la densidad del grafo.
# Salida: 
#             -matriz: matriz de adyacencia del grafo generado
###############################################################################
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

##############################################################################
# Nombre: checkUndirectedM
# Funcionalidad: Evalua si el grafo mG es dirigido o no dirigido.
# Argumentos: 
#             -mG: Diccionario de adyacencia del grafo a evaluar.
# Salida: 
#             -True: Si el grafo es no dirigido.
#             -False: Si el grafo es dirigido.
###############################################################################
def checkUndirectedM(mG):
  for i in range(len(mG)):
    for j in range(len(mG)):
      if mG[i][j] != mG[j][i]:
        return False
  return True

##############################################################################
# Nombre: checkUndirectedD
# Funcionalidad: Evalua si el grafo dG es dirigido o no dirigido.
# Argumentos: 
#             -dG: Diccionario de adyacencia del grafo a evaluar.
# Salida: 
#             -True: Si el grafo es no dirigido.
#             -False: Si el grafo es dirigido.
###############################################################################
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

##############################################################################
# Nombre: initCD
# Funcionalidad: Inicializa un CD al valor inicial -1.
# Argumentos: 
#             -N: Tamaño del CD a inicializar.
# Salida: 
#             -array: Conjunto inicializado.
###############################################################################
def initCD(N):
  array = np.ones(N)*-1
  return array

##############################################################################
# Nombre: union
# Funcionalidad: Devuelve el representante del conjunto obtenidocomo la 
# unión de los representados por los índices rep1, rep2 en elCD almacenado 
# en la tabla pS.
# Argumentos: 
#             -rep1: Indice del CD a tratar.
#             -pS: Tabla que almacena el CD.
#             -rep2: Indice del CD a tratar.
# Salida: 
#             -rep1: Representante del indice indicado.
###############################################################################
def union(rep1, rep2, pS):
  pS[rep2] = rep1 
  return rep1
  
##############################################################################
# Nombre: find
# Funcionalidad: Devuelve el representante del índice ind en el CD almacenado
# en la tabla pS sin realizar o realizando compresión de caminos según flagCC 
# sea False o no.
# Argumentos: 
#             -ind: Indice del CD a tratar.
#             -pS: Tabla que almacena el CD.
#             -flagCC: Bandera para indicar si se desea realizar compresión de 
#             caminos o no.
# Salida: 
#             -z: Representante del indice indicado.
############################################################################### 
def find(ind, pS, flagCC):
  if flagCC == False:
    while pS[ind] != -1:
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

##############################################################################
# Nombre: insertPQ
# Funcionalidad: Inserta en cola de prioridad Q las ramas (u, v) del grafo 
# no dirigido dG para las que u < v.
# Argumentos: 
#             -dG: diccionario de adyacencia del grafo a tratar.
#             -Q: Cola de prioridad para inserción de ramas.
# Salida: 
#             -
###############################################################################
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

##############################################################################
# Nombre: formatToDicc
# Funcionalidad: Función auxiliar para pasar del formato de L(tuplas triples
# (coste, nodou, nodov)) al formato dG de diccionario de adyacencia.
# Argumentos: 
#             -L: Arbol abarcador mínimo en formato de trupla triple.
#             -N: Tamaño del grafo original
# Salida: 
#             -dG: Representación como diccionario de adyacencia del árbol abarcador 
#             mínimo.
###############################################################################
def formatToDicc(L, N):
  dG = dict()
  for i in range(N):
    dG.update({i : []})
  for i in L:
    dG[i[1]].append((i[0],i[2]))
    dG[i[2]].append((i[0],i[1]))
  return dG

##############################################################################
# Nombre: kruskal
# Funcionalidad: Ejecuta el algorimo Krukal sobre el grafo repre-
# sentado por su diccionario de adyacencia.
# Argumentos: 
#             -dG: diccionario de adyacencia del grafo a tratar.
#             -flagCC: Bandera para indicar si se desea realizar compresión de 
#             caminos o no.
# Salida: 
#             -dG: Representación como diccionario de adyacencia del árbol abarcador 
#             mínimo.
###############################################################################
def kruskal(dG, flagCC=True):
  L = [] 
  Q = pq.PriorityQueue()  
  insertPQ(dG, Q)
  S = initCD(len(dG.keys())) 
  while not Q.empty(): 
    w = Q.get() 
    x = find(w[1], S, flagCC)
    y = find(w[2], S, flagCC) 
    if x != y:
      L.append((w[0],w[1],w[2]))
      union(x, y, S) 
  dG = formatToDicc(L,len(dG))
  return dG

udicc = dict()

##############################################################################
# Nombre: timeKruskal
# Funcionalidad: Ejecuta el algorimo codicioso Krukal sobre NGraphs grafos, 
# con tamaño variable entre nNodesIni y nNodesFin de step en step, con un
# sparseFactor dado, y mide los tiempos de ejecución del algoritmo en cada una
# de sus iteraciones.
# Argumentos: 
#             -nGraphs: número de grafos sobre los operar el algortimo.
#             -nNodesIni: Número de nodos mínimo.
#             -nNodesFin: Número de nodos máximo.
#             -step: Numero de nodos de diferencia entre una iteración y otra.
#             -sparseFactor: Factor de dispersión para la generación de grafos.
#             -flagCC: Bandera que marca el uso o no de compresión de caminos.
# Salida: 
#             -timeList: lista de tiempos de ejecución de las iteraciones.
###############################################################################
def timeKruskal(nGraphs, nNodesIni, nNodesFin, step, sparseFactor, flagCC):
  timeList = []
  acum = 0
  i = 0
  for nNodes in range(nNodesIni, nNodesFin, step):
    if i == nGraphs:
      return timeList
    mG = randMatrUndPosWGraph(nNodes, 0.5, maxWeight= 50)
    udicc = fromAdjM2Dict(mG)
    start_time = timeit.default_timer()
      
    #acum += timeit.timeit("test.kruskal(test.udicc)", setup=setup, number = 1)
    kruskal(udicc)
    elapsed = timeit.default_timer() - start_time
    timeList.append(elapsed)
    i = i + 1

  return timeList

##############################################################################
# Nombre: kruskal02
# Funcionalidad: Ejecuta el algorimo Krukal sobre el grafo repre-
# sentado por su diccionario de adyacencia, y mide los tiempos de ejecución del
# cálculo del algortimo obviando las operaciones de colas.
# Argumentos: 
#             -dG: diccionario de adyacencia del grafo a tratar.
#             -flagCC: Bandera para indicar si se desea realizar compresión de 
#             caminos o no.
# Salida: 
#             -dG: Representación como diccionario de adyacencia del árbol abarcador 
#             mínimo.
#             -elapsed: Lista de tiempos de las distintas ejecuciones del algoritmo.
###############################################################################
def kruskal02(dG, flagCC=True):
  timeList = []
  L = [] 
  Q = pq.PriorityQueue()  
  insertPQ(dG, Q)
  S = initCD(len(dG.keys())) 
  #Timer
  start_time = timeit.default_timer()
  while not Q.empty(): 
    w = Q.get() 
    x = find(w[1], S, flagCC)
    y = find(w[2], S, flagCC) 
    if x != y:
      L.append((w[0],w[1],w[2]))
      union(x, y, S) 
  elapsed = timeit.default_timer() - start_time
  dG = formatToDicc(L,len(dG))
  return dG, elapsed

##############################################################################
# Nombre: timeKruskal02
# Funcionalidad: Ejecuta el algorimo codicioso Krukal sobre NGraphs grafos, 
# con tamaño variable entre nNodesIni y nNodesFin de step en step, con un
# sparseFactor dado, y mide los tiempos de ejecución del algoritmo en cada una
# de sus iteraciones, obviando el tiempo de operaciones de cola.
# Argumentos: 
#             -nGraphs: número de grafos sobre los operar el algortimo.
#             -nNodesIni: Número de nodos mínimo.
#             -nNodesFin: Número de nodos máximo.
#             -step: Numero de nodos de diferencia entre una iteración y otra.
#             -sparseFactor: Factor de dispersión para la generación de grafos.
#             -flagCC: Bandera que marca el uso o no de compresión de caminos.
# Salida: 
#             -timeList: lista de tiempos de ejecución de las iteraciones.
###############################################################################
def timeKruskal02(nGraphs, nNodesIni, nNodesFin, step, sparseFactor, flagCC):
  timeList = []
  acum = 0
  i = 0
  for nNodes in range(nNodesIni, nNodesFin, step):
    if i == nGraphs:
      return timeList
    mG = randMatrUndPosWGraph(nNodes, 0.5, maxWeight= 50)
    udicc = fromAdjM2Dict(mG)
    dG, time = kruskal02(udicc, flagCC)
    i = i + 1
    timeList.append(time)
  return timeList


##############################################################################
# Nombre: plotKruskal
# Funcionalidad: Realiza una gráfica aproximando los resultados en tiempos 
# obtenidos en lT con la función matemática func2fit.
# Argumentos: 
#             -lT: lista de tiempos de ejecución del algortimo a representar.
#             -func2fit: Función matemática de aproximación para la gráfica.
#             -nNodesIni: Número de nodos mínimo.
#             -nNodesFin: Número de nodos máximo.
#             -step: Numero de nodos de diferencia entre una iteración y otra.
#   
# Salida: 
#             -timeList: lista de tiempos de ejecución de las iteraciones.
###############################################################################
def plotKruskal(lT, func2fit, nNodesIni, nNodesFin, step):
  n2log = func2fit(nNodesIni, nNodesFin, step);
  print "medidas"
  print lT 
  print len(lT)
  print n2log
  print len(n2log)
  N = len(lT) 

  lr = linear_model.LinearRegression()
  lr.fit(n2log[:, np.newaxis], lT)
  fit = lr.predict(n2log[:, np.newaxis])

  plt.plot(fit)
  plt.plot(lT)
  plt.show()

  return
###############################################################################
# Nombre: incAdy
# Funcionalidad: Devolver la lista de adyacencia e incidencia de un grado dado.
# Argumentos: 
#             -dG: Diccionario de adyacencia de un grafo dirigido.
# Salida: 
#             -Lista de adyacencia de todos los nodos del grafo.
#             -Lista de incidencia de todos los nodos del grafo
###############################################################################
def incAdy(dG):
  lAdy = np.zeros(len(dG))
  lInc = np.zeros(len(dG))

  for k in dG:
    for le in dG[k]:
      lAdy[k] += 1
      lAdy[le[1]] += 1
      lInc[le[1]] += 1
  return lAdy, lInc

###############################################################################
# Nombre: drBP
# Funcionalidad: Función contenedora de la función que desarrolla el algoritmo de 
# busqueda en profundidad en grafos dirigidos.
# Argumentos: 
#             -dG: Diccionario de adyacencia de un grafo dirigido.
# Salida: 
#             -prev: Lista de previos del grafo.
#             -fin: Lista de finalización del grafo
#             -desc: Lista de descubrimiento del grafo.
###############################################################################
def drBP(dG):
  prev = initCD(len(dG))
  fin = initCD(len(dG))
  vist = [False] * len(dG)
  desc = initCD(len(dG))

  for k in dG:
    BP(vist, prev, fin, desc, dG, k)

  return desc, fin, prev

###############################################################################
# Nombre: BP
# Funcionalidad: Función que desarrolla el algoritmo de busqueda en profundidad
# en grafos dirigidos.
# Argumentos: 
#             -dG: Diccionario de adyacencia de un grafo dirigido.
#             -v: Lista de nodos visitados del algoritmo.
#             -p: Lista de previos del grafo.
#             -f: Lista de finalización del grafo
#             -d: Lista de descubrimiento del grafo.
#             -u: Nodo sobre el que realizar la iteración actual.
#             
# Salida: 
#             -
###############################################################################
def BP(v, p, f, d, dG, u):
  count = -1
  if v[int(u)] == False:
    v[int(u)] = True
    #Para comprobar el último tiempo(t) usado
    for k in d:
      if k > count:
        count = k   
    for k in f:
      if k > count:
        count = k  
    #Actualización array descubrimiento
    d[u] = count + 1
    print u
    for el in dG[u]:
      if v[el[1]] == False:
        p[el[1]] = u
      BP(v, p, f, d, dG, el[1])
    #Para comprobar el último tiempo(t) usado
    for k in d:
      if k > count:
        count = k 
    for k in f:
      if k > count:
        count = k 
    #Actualización array finalización
    f[u] = count + 1
     
  else:
    if p[int(u)] != -1: 
      BP(v, p, f, d, dG, p[int(u)])

  
  return 
###############################################################################
# Nombre: BPasc
# Funcionalidad: Desarrollo del algortimo de Busqueda en profundidad, adaptado a
# la búsqueda de ramas ascendentes.
# Argumentos: 
#             -dG: Diccionario de adyacencia de un grafo dirigido.
#             -v: Lista de nodos visitados del algoritmo.
#             -p: Lista de previos del grafo.
#             -f: Lista de finalización del grafo
#             -d: Lista de descubrimiento del grafo.
#             -u: Nodo sobre el que realizar la iteración actual.
#             -asc: Lista de ramas ascendentes (u,v)
# Salida: 
#             -
###############################################################################
def BPasc(v, p, f, d, dG, u, asc):
  count = -1
  if v[int(u)] == False:
    v[int(u)] = True
    #Para comprobar el último tiempo(t) usado
    for k in d:
      if k > count:
        count = k   
    for k in f:
      if k > count:
        count = k  
    #Actualización array descubrimiento
    d[u] = count + 1
    for el in dG[u]:
      if v[el[1]] == False:
        p[el[1]] = u
      #Calculo de ramas ascendentes
      rAscendentes(u, asc, p)

      BPasc(v, p, f, d, dG, int(el[1]), asc)
    #Para comprobar el último tiempo(t) usado
    for k in d:
      if k > count:
        count = k 
    for k in f:
      if k > count:
        count = k    
    #Actualización array finalización
    f[u] = count + 1 
  else:
    rAscendentes(u, asc, p)
    if u != 0: 
      BPasc(v, p, f, d, dG, p[int(u)], asc)
  return 

###############################################################################
# Nombre: rAscendentes
# Funcionalidad: Función auxiliar para el cálculo de ramas ascendentes
# Argumentos: 
#             -p: Lista de previos del grafo.
#             -u: Nodo sobre el que realizar la iteración actual.
#             -asc: Lista de ramas ascendentes (u,v)
# Salida: 
#             -
###############################################################################
def rAscendentes(u,asc,p):
  P = u
  cont = 0   
  while P != -1:
    previoAnterior = P
    P = p[P]
    cont += 1
    if cont > 2:
      asc.append((u,int(previoAnterior))) 
  return 

###############################################################################
# Nombre: detectarCiclos
# Funcionalidad: Función auxiliar para la detección de ciclos.
# Argumentos: 
#             -p: Lista de previos del grafo.
#             -dG: Diccionario de adyacencia de un grafo dirigido.
# Salida: 
#             -True: En caso de haber detectado ciclos en el grafo.
#             -False: En caso de no haber detectado ciclos en el grafo.
###############################################################################
def detectarCiclos(p,dG):
  
  for k in dG:
    for el in dG[k]:
      P = k  
      while P != -1:
        P = p[P]
        if P == el[1]:
          return True
      asc =  P != -1
  return False


###############################################################################
# Nombre: drBPasc
# Funcionalidad: Función contenedora de la función que desarrolla el algoritmo de 
# busqueda en profundidad en grafos dirigidos, aplicada a la detección de ramas 
# ascendentes.
# Argumentos: 
#             -dG: Diccionario de adyacencia de un grafo dirigido.
# Salida: 
#             -asc: Lista de ramas ascendentes.
###############################################################################
def drBPasc(dG):
  prev = initCD(len(dG))
  fin = initCD(len(dG))
  vist = [False] * len(dG)
  desc = initCD(len(dG))
  asc = []

  for k in dG:
    BPasc(vist, prev, fin, desc, dG, k, asc)

  asc = np.unique(asc)
  return asc

###############################################################################
# Nombre: DAG
# Funcionalidad: Función para la detección de ciclos.
# Argumentos: 
#             -dG: Diccionario de adyacencia de un grafo dirigido.
# Salida: 
#             -True: En caso de haber detectado ciclos en el grafo.
#             -False: En caso de no haber detectado ciclos en el grafo.
###############################################################################
def DAG(dG):
  d,f,p = drBP(dG)
  return detectarCiclos(p,dG)

###############################################################################
# Nombre: OT
# Funcionalidad: Función que comprueba si un grafo es DAG y en caso de serlo 
# realiza la ordenación topológica del grafo.
# Argumentos: 
#             -dG: Diccionario de adyacencia de un grafo dirigido.
# Salida: 
#             -OT: Lista con la ordenación topológica del grafo dado.
###############################################################################
def OT(dG):
  OT = []
  q = pq.PriorityQueue()

  #Comprobar si es dirigido
  mG = fromDict2AdjM(dG)
  if checkUndirectedM(mG) == False:
    #Comprobar si tiene ciclos
    if DAG(dG) == False:
      #Realizar ordenacion topologica
      #Comprobar cual es el nodo inicial(menor incidencia)
      ady,inc = incAdy(dG)
      min = np.amin(inc)
      nodo = 0
      for nodo in range(len(inc)):
        if min == inc[nodo]:
          break
      
      #Realizar BP empexando en nodo con menor incidencia
      d,f,p = drBPOT(dG, nodo)

      for k in range(len(f)):
        q.put((f[k], k)) 
      while not q.empty():
        w = q.get()
        OT.append(w)
      OT = OT[::-1]
      for el in range(len(OT)):
        OT[el] = OT[el][1]
      return OT
  return OT
###############################################################################
# Nombre: drBPOT
# Funcionalidad: Función que comprueba si un grafo es DAG y en caso de serlo 
# realiza la ordenación topológica del grafo.
# Argumentos: 
#             -dG: Diccionario de adyacencia de un grafo dirigido.
#             -nodo: Nodo con la menor incidencia por la que empezar el recorrido
#             por el grafo.
# Salida: 
#             -prev: Lista de previos del grafo.
#             -fin: Lista de finalización del grafo
#             -desc: Lista de descubrimiento del grafo.
###############################################################################
def drBPOT(dG, nodo):
  prev = initCD(len(dG))
  fin = initCD(len(dG))
  vist = [False] * len(dG)
  desc = initCD(len(dG))
  
  BP(vist, prev, fin, desc, dG, nodo)
  for n in range(len(vist)):
    if vist[n] == False:
      BP(vist, prev, fin, desc, dG, n)

  return desc, fin, prev

###############################################################################
# Nombre: distMinSingleSourceDAG
# Funcionalidad: Función que mediante ordenación topológica, computa las ditan-
# cias mínimas entre un nodo inicial(source) y el resto de nodos.
# Argumentos: 
#             -dG: Diccionario de adyacencia de un grafo dirigido.
# Salida: 
#             -d: Lista de distancias mínimas entre el nodo inicial y el resto
#             de nodos.
#             -p: Lista de nodos previos para la realización de rutas mínimas.
###############################################################################
def distMinSingleSourceDAG(dG):
  #Comprobar si es dirigido
  mG = fromDict2AdjM(dG)
  if checkUndirectedM(mG) == False:
    #Comprobar si tiene ciclos
    if DAG(dG) == False:
      #Comprobar si tiene una unica fuente
      d,f,p = drBP(dG)
      count = 0
      for prev in p:
        if prev == -1:
          count += 1
      print "fuentes"
      print count
      if count == 1:
        print "Cumplidas condiciones previas"
        #DAG con una única fuente.
        #Comprobar si hay más de un vértice con incidencia 0
        ady,inc = incAdy(dG)
        count = 0
        print "incidencia"
        print inc
        for l in range(len(inc)):
          if inc[l] == 0:
            count += 1
            if count == 1:
              source = l
        if count > 1:
          print "Existe más de un vértice de incidencia 0"
          print "Se usará el vértice %d como source" % source

        #Cálculo de distancias mínimas.
        #Preparación variables
        d = []
        p = []
        L = []
        for i in range(len(dG)):
          p.append(-1) 
          d.append(np.inf) 
        #Obtener la ordenación topológica.
        L = OT(dG)
        print "ordenacion topológica"
        
        d[L[0]] = 0
        for i in range(len(L)):
          for el in dG[L[i]]:
            if d[el[1]] > d[L[i]] + el[0]:
              d[el[1]] = d[L[i]] + el[0]
              p[el[1]] = L[i]
        return d,p
        #Iterar sobre la ordenación topológica para encontrar caminos mínimos
        
      else:
        print "Mas de un nodo fuente encontrado"
    else:
      print "Grafo con ciclos"
  else:
    print "Grafo no dirigido"

  return


