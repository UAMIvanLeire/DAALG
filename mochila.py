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
# Nombre: genSuperCrec
# Funcionalidad: Genera una sucesión supercreciente con n_terms números.
# Argumentos: 
#             -n_terms: Número elementos de la sucesión a generar.
# Salida: 
#             -sucension: Sucesión supercreciente.
###############################################################################

def genSuperCrec(n_terms):
	sucesion = [];
	min = 0;
	for i in range(n_terms):
		sucesion.append(np.random.randint(min, min+100))
		min = np.sum(sucesion)
	return sucesion

###############################################################################
# Nombre: multiplier
# Funcionalidad: Encuentra un número primo relativo con mod y un número generado
# aleatoriamente y que sea superior a mult_ini.
# Argumentos: 
#             -mod: módulo.
#			  -mult_ini: número mínimo para evitar multiplicadores pequeños.
# Salida: 
#             -prim: Entero primo relativo a mod y mult_ini.
###############################################################################

def multiplier(mod, mult_ini):
	prim = np.random.randint(mult_ini, mod)
	while mcd(mod, prim) != 1:
		prim = np.random.randint(mult_ini, mod)
	return prim 



###############################################################################
# Nombre: mcd
# Funcionalidad: Implementación del algoritmo de Euclides para hallar el máximo
# común divisor entre x e y.
# Argumentos: 
#             -x: Elemento del par para el que hallar el máximo común divisor.
#			  -y: Elemento del par para el que hallar el máximo común divisor.
# Salida: 
#             -max: Máximo común divisor.
###############################################################################

def mcd(x, y):
	if x % y == 0:
		return y
	else:
		return mcd(y, x%y)

###############################################################################
# Nombre: inverse
# Funcionalidad: Calcula el inverso de p módulo mod
# Argumentos: 
#             -p: Elemento del par para el que hallar el inverso del módulo.
#			  -mod: Elemento del par para el que hallar el inverso del módulo.
# Salida: 
#             -inv: Inverso del módulo entre p y mod.
###############################################################################

def inverse(p, mod):
	b = 0
	x = 0
	while b < mod:
		x = (p * b) % mod
		if x == 1:
			return b
		b += 1

###############################################################################
# Nombre: modMultInv
# Funcionalidad: Calcular un módulo inverso y multiplicador para la lista super
# creciente dada.
# Argumentos: 
#             -lSC: Sucensión supercreciente. 
# Salida: 
#             -mul: Multiplicador.
#			  -inv: Inverso.
# 			  -mod: Módulo.
###############################################################################

def modMultInv(lSC):
	mod = np.random.randint(np.sum(lSC))
	mul = multiplier(mod, 10)
	inv = inverse(mul, mod)
	return  mul, inv, mod

###############################################################################
# Nombre: genSucesionPublica
# Funcionalidad: Genera la solución pública asociada a la sucesión supercreciente
# , al multiplicador, y al módulo dados.
# Argumentos: 
#             -lSC: Sucensión supercreciente.
#			  -p: Multiplicador.
#			  -mod: Módulo. 
# Salida: 
#             -sol: Solución pública.
###############################################################################

def genSucesionPublica(lSC, p, mod):
	sol = []
	for k in lSC:
		sol.append(k*p%mod)
	return sol

###############################################################################
# Nombre: lPub_2_lSC
# Funcionalidad: Genera la solución privada asociada a la sucesión supercreciente
# , al multiplicador, y al módulo dados.
# Argumentos: 
#             -lSC: Sucensión supercreciente.
#			  -p: Multiplicador.
#			  -mod: Módulo. 
# Salida: 
#             -sol: Solución pública.
###############################################################################

def lPub_2_lSC(l_pub, q, mod):
	sol = []
	for k in l_pub:
		sol.append(k*q%mod)
	return sol

###############################################################################
# Nombre: genRandomBitString
# Funcionalidad: Genera un array de bits aleatorios.
# Argumentos: 
#             -n_bits: tamaño del array de bits.
# Salida: 
#             -array: Array de bits.
###############################################################################

def genRandomBitString(n_bits):
	return np.random.randint(2, size=n_bits)

###############################################################################
# Nombre: MH_encrypt
# Funcionalidad: Genera la lista con el cifrado de cada bloque de la lista de
# bits.
# Argumentos: 
#             -s: Array de bits a cifrar.
#             -lPub: Solución pública.
#             -mod: módulo necesario para el cifrado.
# Salida: 
#             -cifrado: lista con el cifrado de cada bloque.
###############################################################################

def MH_encrypt(s, lPub, mod):
	tamMod = len(s)%len(lPub)
	
	lPriv = lPub_2_lSC(lPub, q, mod)
	#Rellenamos con 0 para conseguir un tamaño multiplo de lPub.
	if tamMod != 0:
		zero = np.zeros((len(lPub)-tamMod,), dtype=np.int)
		s = np.concatenate((s, zero), axis=0)
	#Transforammos array en matriz MxN
	sArray = np.array(s)
	matrix = np.reshape(sArray, (-1, len(lPub)))
	print matrix
	#Transformamos cada bloque en su entero para poder cifrarlo.
	for k in matrix:
		sum = 0
		for i in range(len(lPub)):
			if k[i] == 1:
				sum += lPriv[i]

		# str1 = ''.join(str(e) for e in k)
		# num = int(str1, 2)
		
	return 


#TESTS 

lSC = genSuperCrec(10)
print "lSC"
print lSC
print mcd(65, 15)
print multiplier(30, 10)
print inverse(117, 244)
mod,mul,inv = modMultInv(lSC)
print "mod"
print mod
print "mul"
print mul
print "inv"
print inv
print "solPublica"
solPublica = genSucesionPublica(lSC, mul, mod)
print solPublica
print "solPrivada"
solPrivada = lPub_2_lSC(solPublica, inv, mod)
print solPrivada
print "array de bits"
bits = genRandomBitString(16)
print bits
print "Cifrado"
MH_encrypt(bits, [1,2,3,4,5], 6)