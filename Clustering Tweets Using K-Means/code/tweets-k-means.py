import sys
import json
import re

class tweet_k_means:
    
    def __init__(self, tweets_data,seed_file,no_of_clusters):
        with open(tweets_data) as json_data:
            data=json_data.read()
        json_data=re.split('\n',data)
        self.data={}
        for data in json_data:
            self.data[json.loads(data)["id"]]=json.loads(data)["text"]
        with open(seed_file) as seed:
            data=seed.read()
        self.seed_data=re.split(',\n',data)
        self.K=no_of_clusters
        
    def calculate_centroids(self):
        cluster_centroid=[]
        while(1):
            same=1
            for k in self.seed_data:
                if k not in cluster_centroid:
                    same=0
                    break
            if same==1:
                break
            cluster_centroid=self.seed_data
            self.form_cluster()
            self.re_seed()
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
        for data_point in self.data:
            min_distance=1
            nearest_point=0
            for centroid in self.seed_data:
                distance=self.calculte_distance(self.data[data_point],self.data[long(centroid)])
                if(distance<min_distance):
                    min_distance=distance
                    nearest_point=centroid
            self.cluster_list[nearest_point].append(data_point)
    
    def calculte_distance(self,data_point1,data_point2):
        data_point1_list= re.split('\n|,|\t|\s',data_point1)
        data_point2_list= re.split('\n|,|\t|\s',data_point2)
        union=self.union(data_point1_list,data_point2_list)
        intersection=self.intersection(data_point1_list,data_point2_list)
        distance=1-float(intersection)/union
       
        return distance
    
    def re_seed(self):
        self.seed_data=[]
        for cluster in self.cluster_list:
            min_avg_distance=1
            centroid=cluster
            for test_point in self.cluster_list[cluster]:
                distance=0
                for point in self.cluster_list[cluster]:
                    distance+=self.calculte_distance(self.data[point],self.data[test_point])
                avg_distance=float(distance)/len(self.cluster_list[cluster])
                if(avg_distance<min_avg_distance):
                    centroid=test_point
                    min_avg_distance=avg_distance
            self.seed_data.append(centroid)
    
    def Calculate_SSE(self):
        self.SSE=0
        for cluster in self.cluster_list:
            for point in self.cluster_list[cluster]:
                distance=self.calculte_distance(self.data[point],self.data[long(cluster)])
                self.SSE+=distance*distance
                
    def union(self,data_point1_list,data_point2_list):
        union_list=[]
        for data in data_point1_list:
            if(data not in union_list):
                union_list.append(data)
        for data in data_point2_list:
            if(data not in union_list):
                union_list.append(data)
        return len(union_list)

    def intersection(self,data_point1_list,data_point2_list):
        intersection_list=[]
        for data in data_point1_list:
            if(data in data_point2_list):
                intersection_list.append(data)
        return len(intersection_list)
        
        
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
        seed_file=sys.argv[2]
        tweets_data=sys.argv[3]
        output_file=sys.argv[4]
     except:
        print ""
        print "Usage: tweets-k-means.py <numberOfClusters> <initialSeedsFile> <TweetsDataFile> <outputFile>"
        exit()
     k_means=tweet_k_means(tweets_data,seed_file,no_of_clusters)
     k_means.calculate_centroids()
     k_means.output_clusters(output_file)