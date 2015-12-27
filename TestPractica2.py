#TEST

# m = np.zeros(shape = (4, 4))
# for i in range (4): m[i] = np.inf

# m[0][1] = 2
# m[1][0] = 3
# m[1][2] = 1
# m[2][1] = 4
# m[2][3] = 2
# m[3][1] = 1

# print m 

# dicc = fromAdjM2Dict(m)
# print dicc

# checkUndirectedD(dicc)

# #Prueba diccionarios no dirigidos

# mG = randMatrUndPosWGraph(4, 0.5, maxWeight= 50)

# udicc = fromAdjM2Dict(mG)
# print udicc
# print checkUndirectedM(mG)
# print checkUndirectedD(udicc)
# print checkUndirectedD(dicc)


#Test Kruskal
# Q = pq.PriorityQueue()
# insertPQ(udicc, Q)
# while not Q.empty():
#   print Q.get()
udicc2 = {0:[(10,1), (12,2), (5,1)], 1:[(10,0),(4,3),(5,0),(6,2)], 2:[(12,0),(6,1),(8,3)], 3:[(4,1),(8,2)]}
udicc3 = {0: [(6.0, 1), (37.0, 2),(15,3)], 1: [(6.0, 0), (40.0, 2), (40.0, 3)], 2: [(37.0, 0), (40.0, 1)], 3: [(40.0, 1),(15,0)]}
dicc3 = {0: [(6.0, 1)], 1: [(40.0, 2), (40.0, 3)], 2: [(37.0, 0)], 3: [(15,0)]}
diccInconexo = {0: [(6.0, 1)], 1: [(40.0, 2), (40.0, 3)], 2: [(37.0, 0)], 3: [(15,0)], 4:[(10,1)]}
diccInconexo2 = {0: [(6, 1)], 1: [(40, 2), (40, 3)], 2: [(37, 0)], 3: [(15,0)], 4:[(12,1)], 5:[(30,0), (25,3)]}
diccAscAciclico = {0: [(3,1), (4,2), (3,4)], 1: [(3,3)], 2: [], 3: [(5,2)], 4:[(9,3)]}
diccAscAciclico2 = {0: [(3,1), (4,2), (3,4)], 1: [(3,3)], 2: [], 3: [(5,2)], 4:[(1,3), (6,5)], 5:[]}
diccAscCiclico = {0: [(3,1), (4,2), (3,4)], 1: [(3,3)], 2: [], 3: [(5,2)], 4:[(9,3), (6,5)], 5:[(3,0)]}
diccAscAciclico3 = {0: [(3,1), (4,2), (3,4)], 1: [(3,3)], 2: [], 3: [(5,2)], 4:[(9,3)], 5:[(5,0)], 6:[(2,0)]}
# print checkUndirectedD(udicc3)
# print udicc2
# print "KRUSKAL"
# print kruskal(udicc2, flagCC=False)

# print "timeKruskal"
# time = timeKruskal(10, 1, 1000, 10, 0.5, True)
# print time
# print timeKruskal102(10, 1, 1000, 10, 0.5, True)

# plotKruskal(time, n2log, 1, 100, 10)

#TEST BP

# a,b = incAdy(dicc3)
# print a
# print b

# d,f,p = drBP(diccAscAciclico2)
# print "previos"
# print p
# print "finalización"
# print f
# print "descubrimiento"
# print d

# a = drBPasc(diccAscAciclico2)

# print "ascendentes"
# print a

# print DAG(diccAscCiclico)

# print "OT"
# print OT(diccAscAciclico2)

# print distMinSingleSourceDAG(diccAscAciclico2)
# print TGF2dG("dg.tgf")
# print fromDict2AdjM(udicc3)





