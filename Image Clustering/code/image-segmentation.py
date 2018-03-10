# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 11:51:32 2017

@author: amal
"""

import sys
import cv2
import numpy as np
import sys
import pandas as pd
import random

class img_seg:
    
    def __init__(self, input_image,no_of_clusters):
        self.image = cv2.imread(input_image)
        self.K=no_of_clusters
            
    def find_max_min(self):
        max_x= -1000
        max_y= -1000
        max_z= -1000
        min_x=1000
        min_y=1000
        min_z=1000        
        for height in range(0,len(self.image)):
            for width in range(0,len(self.image[height])):
                x=self.image[height][width][0]
                y=self.image[height][width][1]
                z=self.image[height][width][2]
                if(min_x>x):
                    min_x=x
                if(min_y>y):
                    min_y=y
                if(min_z>z):
                    min_z=z
                if(max_x<x):
                    max_x=x
                if(max_y<y):
                    max_y=y
                if(max_z<z):
                    max_z=z
        return[min_x,min_y,min_z,max_x,max_y,max_z]
    
    def initial_seed(self,max_min):
        self.seed_data=[]
        for i in range(0,int(self.K)):
            x=random.uniform(max_min[0],max_min[3])
            y=random.uniform(max_min[1],max_min[4])
            z=random.uniform(max_min[2],max_min[5])
            if((x,y,z) not in self.seed_data):
                self.seed_data.append((x,y,z))
    
    def calculate_centroids(self):
        max_min=[0,0,0,255,255,255]
        self.initial_seed(max_min)
        cluster_centroid=[]
        for i in range(0,25):
            same=1
            for k in self.seed_data:
                if k not in cluster_centroid:
                    same=0
                    break
            if same==1:
                break
            cluster_centroid=self.seed_data
            self.form_cluster()
            self.re_seed(max_min)
        
        self.quantize()
                
    def quantize(self):
        for cluster in self.cluster_list:
            for points in self.cluster_list[cluster]:
                self.image[points[0]][points[1]]=list(cluster)
                
    def form_cluster(self):
        self.cluster_list={}
        for i in self.seed_data:
            self.cluster_list[i]=[]
      
        for height in range(0,len(self.image)):
            for width in range(0,len(self.image[height])):            
                min_distance=1000
                nearest_point=0
                for centroid in self.seed_data:
                    distance=self.calculate_distance(self.image[height][width],centroid)
                    if(distance<min_distance):
                        min_distance=distance
                        nearest_point=centroid
                self.cluster_list[nearest_point].append((height,width))
    
    def calculate_distance(self,data_point1,data_point2):
        distance=abs(float(data_point1[0])-float(data_point2[0]))+abs(float(data_point1[1])-float(data_point2[1]))+abs(float(data_point1[2])-float(data_point2[2]))
       
        return distance
    
    def re_seed(self,max_min):
        self.seed_data=[]
        for cluster in self.cluster_list:
            x=0
            y=0
            z=0
            for points in self.cluster_list[cluster]:
                x+=self.image[points[0]][points[1]][0]
                y+=self.image[points[0]][points[1]][1]
                z+=self.image[points[0]][points[1]][2]
            if(len(self.cluster_list[cluster])==0):
                x=random.uniform(max_min[0],max_min[3])
                y=random.uniform(max_min[1],max_min[4])
                z=random.uniform(max_min[2],max_min[5])
            else:
                x=float(x)/len(self.cluster_list[cluster])
                y=float(y)/len(self.cluster_list[cluster])
                z=float(z)/len(self.cluster_list[cluster])
            self.seed_data.append((x,y,z))
    
    def save_cluster(self,output_file):
         cv2.imwrite( output_file, self.image );
        
if __name__=='__main__':
    try:
        no_of_clusters=sys.argv[1]
        input_image=sys.argv[2]
        output_image=sys.argv[3]
    except:
        print ""
        print "Usage: km.py <numberOfClusters> <inputImage> <outputImage>"
        exit() 
    k_mean=img_seg(input_image,no_of_clusters)
    k_mean.calculate_centroids()
    k_mean.save_cluster(output_image)
