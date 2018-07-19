# This file is used to test clustering algorithm Learning Vector Quantization (LVQ).

# Note : x1 means density, while x2 means sweet degree, meanwhile y means label.
# Label Info : there are two classes in this dataset. However, we want to divide the data into five clusters.

# File Name : clustering.py 
# Copyright : C.Ma 2016

from __future__ import division
from scipy.spatial import Voronoi,voronoi_plot_2d
import numpy as np
import matplotlib.pyplot as plt
import random

##########################################################
# Function Time

# This Function is used to draw data graph
def data_graph(x1,x2,p1,p2,iter) :
	points = [[p1[0],p2[0]],[p1[1],p2[1]],[p1[2],p2[2]],[p1[3],p2[3]],[p1[4],p2[4]]]
	if iter == 0 :
		title = "Initialization Graph"
	else :
		title = "After " + str(iter) + " iterations"
	
	# compute Voronoi tesselation
	vor = Voronoi(points)
	# plot
	voronoi_plot_2d(vor)
	
	plt.plot (x1[8 : 21],x2[8 : 21],'b.',x1[0 : 8],x2[0 : 8],'r*',x1[21 : 30],x2[21 : 30],'r*')
	plt.plot(p1,p2,'ro')
	plt.xlabel("Density")
	plt.ylabel("Sugar Ratio")
	plt.title(title)
	plt.axis([0.2,0.9,0,0.5])

# This Function is used to calculate the distance of two nodes
def distfun(x1,y1,x2,y2) :
	return ((x1 - x2)**2 + (y1 - y2)**2)**0.5

##########################################################

# The detail of the Water Melon Dataset 4.0
x1 = [0.697,0.774,	0.634,	0.608,	0.556,	0.403,	0.481,	0.437,	0.666,	0.243,	0.245,	0.343,	0.639,	0.657,	0.36,	0.593,	0.719,	0.359,	0.339,	0.282,	0.748,	0.714,	0.483,	0.478,	0.525,	0.751,	0.532,	0.473,	0.725,	0.446]

x2 = [0.46,	0.376,	0.264,	0.318,	0.215,	0.237,	0.149,	0.211,	0.091,	0.267,	0.057,	0.099,	0.161,	0.198,	0.37,	0.042,	0.103,	0.188,	0.241,	0.257,	0.232,	0.346,	0.312,	0.437,	0.369,	0.489,	0.472,	0.376,	0.445,	0.459]

# y means the label of watermelon
y = [0,	0,	0,	0,	0,	0,	0,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0]

# We use this figure to show the origin data info
#plt.figure('Water Melon Dataset 4.0 Info')
# this sentence is used to show the label info
#plt.plot (x1[8 : 21],x2[8 : 21],'bo',x1[0 : 8],x2[0 : 8],'r*',x1[21 : 30],x2[21 : 30],'r*')
#plt.plot(x1,x2,'ro')
#plt.xlabel("Density")
#plt.ylabel("The Sweet of Water Melon")
#plt.title("Data Graph")

#plt.show()

# Initialization of learning vector
p1 = [0,0,0,0,0]
p2 = [0,0,0,0,0]
x1_max = max(x1)
x1_min = min(x1)
x2_max = max(x2)
x2_min = min(x2)

for i in range(0,5) :
	p1[i] = random.uniform(x1_min,x1_max)
	p2[i] = random.uniform(x2_min,x2_max)

# t means the label of learning vector
t = [0,0,0,1,1]
# alpha means learing ratio, where it is written as elta
alpha = 0.1 

# Here, we set a iteration of 500, you can change the iteration structure.	
for iter in range(0,500) :	
	if iter == 0 :
		data_graph(x1,x2,p1,p2,iter)
	elif iter == 49 :
		data_graph(x1,x2,p1,p2,iter+1)
	elif iter == 99 :
		data_graph(x1,x2,p1,p2,iter+1)
	elif iter == 199 :
		data_graph(x1,x2,p1,p2,iter+1)
	elif iter == 499 :
		data_graph(x1,x2,p1,p2,iter+1)

	j = random.randint(0,29)
	flag = -1
	dist_min = -1
	label_flag = -1
	
	for i in range(0,5) :
		dist = distfun(p1[i],p2[i],x1[j],x2[j])
		if i == 0 :
			dist_min = dist
			flag = i
			continue
		if dist_min > dist :
			dist_min = dist
			flag = i
			
	if y[j] == t[flag] :
		p1[flag] += alpha * ( x1[j] - p1[flag] )
		p2[flag] += alpha * ( x2[j] - p2[flag] )
		label_flag = 1
	else :
		p1[flag] -= alpha * ( x1[j] - p1[flag] )
		p2[flag] -= alpha * ( x2[j] - p2[flag] ) 
		label_flag = 0
	
	# This part shows the details of the iteration 
	# The Reason why I write this is that it is easily for us to see the Convergence or not	
	# In order to make your terminal clear, you don't need to run these codes until necessary	
	#print "iter is:",iter
	#print "flag is:",flag
	#print "Label Flag is:",label_flag
	#print "the change of p:",p1[flag],p2[flag]
	#print "---------------------------------"

plt.show()
print("Copyright : C.Ma 2016")
print("All Done!")