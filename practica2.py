import numpy as np
import numpy.ma as ma

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
	for clave, valor in dG.iteritems():
		print valor
		while valor: 
			#l = valor.pop()
			#print l
			print "hola"
	return True

#Pruebas parte 1

mG = randMatrUndPosWGraph(4, 0.5, maxWeight= 50.)
print mG
print checkUndirectedM(mG)

dG = {0: {1: 7.0}, 1: {0: 27.0}, 2: {}, 3: {0: 24.0, 2: 5.0}}
print dG

checkUndirectedD(dG)

# m = np.zeros(shape = (4, 4))
# for i in range (4): m[i] = np.inf

# m[0][1] = 2
# m[1][0] = 2
# m[1][2] = 4
# m[2][1] = 6
# m[2][3] = 3
# m[3][1] = 2
# mG = m
# print m

# print checkUndirectedM(m)
