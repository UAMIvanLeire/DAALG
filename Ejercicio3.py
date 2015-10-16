import numpy as np
import Queue as pq

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



def dijkstra():

	return 

def timeDijkstraM(nGraphs, nNodesIni, nNodesFin, step, sparseFactor=.25): 

	return 
	



mG = randMatrPosWGraph(4, 0.5, maxWeight= 50)

m = np.zeros(shape = (4, 4))
for i in range (4): m[i] = np.inf

m[0][1] = 2
m[1][0] = 3
m[1][2] = 1
m[2][1] = 4
m[2][3] = 2
m[3][1] = 1

print m 



print cuentaRamas(mG)
print fromAdjM2Dict(mG)
print fromDict2AdjM(fromAdjM2Dict(mG))
print dijkstraM(m,3)

print fromAdjM2Dict(m)