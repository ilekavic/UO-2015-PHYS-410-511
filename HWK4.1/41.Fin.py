# -*- coding: utf-8 -*-
"""
Created on Mon May 04 13:11:40 2015

@author: IL
"""

from math import radians, sin, cos, asin, sqrt
import matplotlib.pyplot as plt

import os
import numpy as np
from pandas import *
from pandas.io.parsers import read_fwf


def dist(lat1,lon1,lat2,lon2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula to calculate the distance between two points on surface on the earth
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 3956 # Radius of earth in kilometers. Use 3956 for miles, 6371 for km
    return c * r

file = "C:/Users/IL/Documents/SciComp/HWK4.1/quake.txt"
data = read_fwf(file,header=None,names=["technique", "year", "lat", "long", "depth", "scale"])

seattleLocation = (47.6,-122.3)

def inRadius(max):
    return data.apply(lambda row: dist(row["lat"],row["long"],seattleLocation[0],seattleLocation[1]) < max,axis=1)

print(data[inRadius(100)])

    
    
#Read in modified quake file, removed 1st column, as well as "r","q", etc.
with open('C:/Users/IL/Documents/SciComp/HWK4.1/quake5.txt') as file:
    qdata = [[float(digit) for digit in line.split()] for line in file]
    
print qdata[1][:]



thresh = 50 #dist returns distance in miles
qArr = np.reshape(qdata,(len(qdata),5))

maxClust = 100       
clus1 = np.zeros((len(qArr),maxClust))

#Find initial clusters
for i in range(0, len(qdata)/10,1):
    ctr = 0
    for j in range(0, len(qdata)/10 ,1):
        if( (qArr[i,0] < qArr[j,0]+2 and qArr[i,0] > qArr[j,0]-2) and dist(qArr[i,1],qArr[i,2],qArr[j,1],qArr[j,2]) < thresh ):
            #Store index 
            
            clus1[i,ctr] = j
            ctr += 1
            
#Iterate through, clus1, if two rows share two elements, combine, delete one
for i in range(0,len(qArr)/10,1):
    for j in range(0,len(qArr)/10,1):
        ctr =0
        if i != j:
            test = []
            if len(test) > 2:
                test.addAll(set(clus1[i,:]) or set(clus1[j,:]))
                k=0
                while k in range(0, len(test),1):
                    if (test[k] != 0):
                        clus1[i,k] = test[k]
                        k += 1
                
#New array, takes the postion of first element in 1st two columns, third column is number of events in cluster
clus2 = np.zeros((len(qArr),3))

for i in range(0,len(qArr),1):
    clus2[i,0]= qArr[clus1[i,0],1]
    clus2[i,1]= qArr[clus1[i,0],2]
    j=0
    while clus1[i,j] != 0:
        j += 1
    clus2[i,2] = j
    
print clus2[10]

N = len(qArr)
colors = np.random.rand(len(clus2))
plt.scatter(clus2[:,1], clus2[:,0],(clus2[:,2]-1)*5,c=colors, alpha =.5)
plt.show()
        
b = qArr[(qArr[:,2] < -75)*(qArr[:,2] > -150),:]
colors2 = np.random.rand(len(b))

plt.scatter(b[:,2],b[:,1], b[:,3], c =colors2, alpha =.5 )
plt.show()


#Timeline of all events mag 7 or greater

ave = np.zeros((2,22))

for n in range(22):
    ctr = 0
    for i in range (len(qArr)):
        if( qArr[i,0] >= 1900 +5*n and qArr[i,0] < 1900 + 5*(n+1) and qArr[i,4]>7):
            ave[0,n] += qArr[i,4]
            ctr +=1
            
    ave[0,n] = ctr
    ave[1,n] = 1900 + 5*n
    
for i in range(22):
    print"'",ave[1,i],"'", ","
    


       

