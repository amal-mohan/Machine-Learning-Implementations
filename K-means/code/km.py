import sys
import random
import re
import math

class k_means:
    
    def __init__(self, input_data,no_of_clusters):
        with open(input_data) as data_txt:
            data=data_txt.read()
        datas=re.split('\r',data)
        self.points={}
        for data in datas:
            point= re.split('\t',data)
            if(point[0]!='id'):
                self.points[point[0]]=((float(point[1]),float(point[2])))
        self.K=no_of_clusters
        
    def find_max_min(self):
        max_x= -1000
        max_y= -1000
        min_x=1000
        min_y=1000
        for point in self.points:
            x=float(self.points[point][0])
            y=float(self.points[point][1])
            if(min_x>x):
                min_x=x
            if(min_y>y):
                min_y=y
            if(max_x<x):
                max_x=x
            if(max_y<y):
                max_y=y
        return[min_x,min_y,max_x,max_y]
    
    def initial_seed(self,max_min):
        self.seed_data=[]
        for i in range(0,int(self.K)):
            x=random.uniform(max_min[0],max_min[2])
            y=random.uniform(max_min[1],max_min[3])
            if((x,y) not in self.seed_data):
                self.seed_data.append((x,y))
    
    def calculate_centroids(self):
        max_min=self.find_max_min()
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
        self.Calculate_SSE()
        c_list={}
        for i in range(0,int(self.K)):
            c_list[i]=[]
        i=0
        for cluster in self.cluster_list:
                for test_point in self.cluster_list[cluster]:
                    c_list[i].append(test_point)
                i+=1
        self.cluster_list=c_list
        
    def form_cluster(self):
        self.cluster_list={}
        for i in self.seed_data:
            self.cluster_list[i]=[]
      
        for data_point in self.points:
            
            min_distance=5
            nearest_point=0
            for centroid in self.seed_data:
                distance=self.calculate_distance(self.points[data_point],centroid)
                if(distance<min_distance):
                    min_distance=distance
                    nearest_point=centroid
            self.cluster_list[nearest_point].append(self.points[data_point])
    
    def calculate_distance(self,data_point1,data_point2):
        distance=math.sqrt((float(data_point1[0])-float(data_point2[0]))*(float(data_point1[0])-float(data_point2[0]))+(float(data_point1[1])-float(data_point2[1]))*(float(data_point1[1])-float(data_point2[1])))       
        return distance
    
    def re_seed(self,max_min):
        self.seed_data=[]
        for cluster in self.cluster_list:
            x=0
            y=0
            for test_point in self.cluster_list[cluster]:
                x+=test_point[0]
                y+=test_point[1]
            if(len(self.cluster_list[cluster])==0):
                x=random.uniform(max_min[0],max_min[2])
                y=random.uniform(max_min[1],max_min[3])
            else:
                x=float(x)/len(self.cluster_list[cluster])
                y=float(y)/len(self.cluster_list[cluster])
            self.seed_data.append((x,y))
    
    def Calculate_SSE(self):
        self.SSE=0
        for cluster in self.cluster_list:
            for point in self.cluster_list[cluster]:
                distance=self.calculate_distance(point,cluster)
                self.SSE+=distance*distance
                
        
    def output_clusters(self,output_file):
        with open(output_file,'w') as f:
            for cluster in self.cluster_list:
                f.write(str(cluster)+" ")
                for test_point in self.cluster_list[cluster]:
                    f.write(str(test_point)+",")
                f.write("\n")
            f.write("\nSSE: "+str(self.SSE))   
        
if __name__=='__main__':
     try:
        no_of_clusters=sys.argv[1]
        input_file=sys.argv[2]
        output_file=sys.argv[3]
     except:
        print ""
        print "Usage: km.py <numberOfClusters> <inputFile> <outputFile>"
        exit()
     k_mean=k_means(input_file,no_of_clusters)
     k_mean.calculate_centroids()
     k_mean.output_clusters(output_file)