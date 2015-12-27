
#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import sys

##### importar archivo grafos.py "bueno"
sys.path.append("D:\Google Drive\Cursos\DAA\DAA 2015 2016\practicas\pract01")
import grafos as gr

sys.path.append(".")
import grafos15 as grp

################################################################ big main
def mainPract2(args):
  if len(args) > 1 and len(args) != 3:
    print ("args: either none or name of tgf files with an undirected and directed grapsh to be reads to check Kruskal and the BP and related functions")
    sys.exit(0)  
  fNameKruskal = fNameBP = ''
  if len(args) == 3:
    fNameKruskal = args[1]; fNameBP = args[2]
  
  ################################ generar, guardar y leer gr no Dirigido
  print("\n################################ generando/leyendo grafos ...\n")

  if fNameKruskal == '':
    nNodes=7; sparseFactor=0.7
    mG = grp.randMatrUndPosWGraph(nNodes, sparseFactor)
    dG = grp.fromAdjM2Dict(mG)
    grp.dG2TGF(dG, 'grnodrand.tgf')
    dGnoD = grp.TGF2dG('grnodrand.tgf')
  else:
    dGnoD = gr.TGF2dG(fNameKruskal)
  print("\tgrafo escrito:\n"); print(dG)
  print("\n\tgrafo leído:\n"); print(dGnoD)
  
  ################################ kruskal y sus tiempos
  ################ kruskal básico
  print("\n################################ Kruskal básico ...\n")

  L = grp.kruskal(dGnoD, flagCC=True)
  print(L)
  cAAm, L = gr.kruskal(dGnoD, flagCC=True)
  print(L)
  
  ################ tiempos kruskal
  print("\n################################ timing Kruskal ...\n")
 
  lTp  = grp.timeKruskal(nGraphs=20, nNodesIni=10, nNodesFin=30, step=10, sparseFactor=0.75, flagCC=True)
  lT   = gr.timeKruskal(nGraphs=20, nNodesIni=10, nNodesFin=30, step=10, sparseFactor=0.75, flagCC=True)
  print "\ttiempos de kruskal:\nTp", lTp, "\nT", lT
  
  ################################ BP básica
  print("\n################################ BP básica ...\n")
 
  if fNameBP == '':
    mG = grp.randMatrPosWGraph(nNodes, sparseFactor)  ########## generar grafo dirigido primero como MdA y luego como GfA
    dG = grp.fromAdjM2Dict(mG)
  else:
    dG = gr.TGF2dG(fNameBP)
  d, f, p = grp.drBP(dG)
  print "d: ", d, "\nf: ", f, "\np: ", p
  
  ################ BP para detectar ramas ascendentes
  print("\n################################ BP para ramas ascendentes ...\n")
 
  lAsc = grp.drBPasc(dG)
  print "ramas asc:\n", lAsc
  
  ################################ OT y dist min en DAGs
  print("\n################################ OT y OT y dist min en DAGs ...\n")
  
  ################ generar DAG
  print("################################ generando DAG ...\n")

  nNodes = 10; sparseFactor = 0.2; nIntentosMax=1000
  flDAG = False; nIntentos = 0
  while not flDAG and nIntentos < nIntentosMax:
    mG = grp.randMatrPosWGraph(nNodes, sparseFactor)
    dG = grp.fromAdjM2Dict(mG)
    flDAG = grp.DAG(dG)
    nIntentos +=1
  print("num intentos hasta DAG: %d" % nIntentos)
  print "DAG:\n", dG
  
  ################ dist min en single source DAGs (sólo si se ha encontrado uno)
  print("\n################################ dist min en single source DAGs ...\n")
  
  if flDAG:
    lOT = grp.OT(dG)
    d, p = grp.distMinSingleSourceDAG(dG)
    print "\nd: ", d, "\np: ", p
    
################################################################ calling main
if __name__ == '__main__':
  '''llamada correcta: fNameKruskal, fNameBP; hacer ambos '' para generación de grafos '''
  mainPract2(sys.argv)  