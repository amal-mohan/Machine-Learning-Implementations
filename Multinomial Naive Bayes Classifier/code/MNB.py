# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 16:39:09 2017

@author: amal
"""

import os
import sys
import re
import math
import random

class NB:
    
    def __init__(self,training_data_path,test_data_path):
        self.training_data=training_data_path
        self.test_data=test_data_path
        self.Select_Data()
        self.stop_words=['a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', "can't", 'cannot', 'could', "couldn't", 'did', "didn't", 'do', 'does', "doesn't", 'doing', "don't", 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', "hadn't", 'has', "hasn't", 'have', "haven't", 'having', 'he', "he'd", "he'll", "he's", 'her', 'here', "here's", 'hers', 'herself', 'him', 'himself', 'his', 'how', "how's", 'i', "i'd", "i'll", "i'm", "i've", 'if', 'in', 'into', 'is', "isn't", 'it', "it's", 'its', 'itself', "let's", 'me', 'more', 'most', "mustn't", 'my', 'myself', 'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'same', "shan't", 'she', "she'd", "she'll", "she's", 'should', "shouldn't", 'so', 'some', 'such', 'than', 'that', "that's", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', "there's", 'these', 'they', "they'd", "they'll", "they're", "they've", 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was', "wasn't", 'we', "we'd", "we'll", "we're", "we've", 'were', "weren't", 'what', "what's", 'when', "when's", 'where', "where's", 'which', 'while', 'who', "who's", 'whom', 'why', "why's", 'with', "won't", 'would', "wouldn't", 'you', "you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself', 'yourselves']
        self.matrix={}
        self.all_word=[]
        self.total_instances=0
        self.mode='train'
        self.train_instances={}
        self.errors=0
        self.total_test_instances=0
        
        for classs in self.classes:
            self.matrix[classs]={}
    
    def Select_Data(self):
        indices=[]
        self.classes=[]
        for i in range(0,5):
            index=random.randint(0,len(os.listdir(self.training_data))-1)
            while(1):
                if index in indices:
                    index=random.randint(0,len(os.listdir(self.training_data))-1)
                else:
                    break
            indices.append(index)
            self.classes.append(os.listdir(self.training_data)[index])
            
      
    
    def Create_Model(self):
        for classs in self.classes:
            if(self.mode=='train'):
                folder_path=os.path.join(self.training_data,classs)
            else:
                folder_path=os.path.join(self.test_data,classs)
            self.current_class=classs
            self.Navigate_Path(folder_path)
     #   print self.matrix
#        print self.all_word
        if self.mode=='train':
            self.Calculate_Test_Parameters()

    
    def Navigate_Path(self,folder_path):
        if self.mode=='train':
            self.train_instances[self.current_class]=len(os.listdir(folder_path))
            self.total_instances+=len(os.listdir(folder_path))
        else:
            self.total_test_instances +=len(os.listdir(folder_path))
        for filenames in os.listdir(folder_path):
            current_file=os.path.join(folder_path,filenames)
            self.Word_Store(current_file)
            
    def Word_Store(self,current_file):
        with open(current_file, 'r') as c_file:
            data=c_file.read()
        split_data=re.split('Lines: \d*',data)
         
        word_string=split_data[len(split_data)-1]
#        if(len(split_data)==1):
#            word_string=split_data[0]
#        elif(len(split_data)==2):
#            word_string=split_data[1]
#        elif(len(split_data)==3):
#            word_string=split_data[2]
        
        word_list=self.Clean_Data(word_string)

        if(self.mode=='train'):
            self.Construct_Matrix(word_list)
        else:
            self.Class_Predictor(word_list)            
    
    def Test_Data(self):
        self.mode='test'
        self.Create_Model()
        
        Accuracy=(float(self.total_test_instances-self.errors)/self.total_test_instances)*100
        return Accuracy
        
    def Calculate_Test_Parameters(self):
        self.unique_words=len(self.all_word)
        self.class_words={}
        for classes in self.classes:
            count=0
            for words in self.matrix[classes]:
                count+=self.matrix[classes][words]
            self.class_words[classes]=count
            
                
        
    def Clean_Data(self,word_string):

        word_list=re.split('\n|,|\t|\s',word_string)
        word_list=list(filter(None,word_list))
        
        remove_char=['.',',','\'','\"','?','!','~','`','$','#','%','*','(',')','^','&','-','_','+','=','|','\\',';',':','/','>','<','[',']','{','}']

        for i in range(0,len(word_list)):
            if(len(word_list[i])==1):
                word_list[i]=""
            else:
                for j in range(0,len(word_list[i])):
                    if(word_list[i][j] not in remove_char):
                        break
                if(j>0):
                    word_list[int(word_list.index(word_list[i]))]=word_list[i][j:]
                if(len(word_list[i])!=0):
                    k=0
                    for j in xrange(len(word_list[i])-1,0,-1):
                        if(word_list[i][j] not in remove_char):
                            break
                        else:
                            k+=1
                    if(k>0):                        
                        word_list[int(word_list.index(word_list[i]))]=word_list[i][:-(k)]
                     
                if(len(word_list[i])==1):
                    word_list[i]=""
                if(word_list[i] in self.stop_words):
                    word_list[i]=""
        word_list=list(filter(None,word_list))
        return word_list
    
    def Class_Predictor(self,word_list):
        word_counter={}
        predicted_class=""
        predicted_value=-1
        for word in word_list:
            if word.lower() in word_counter.keys():
                word_counter[word.lower()]+=1
            else:
                word_counter[word.lower()]=1
        for classs in self.classes:
            probability=math.log(float(self.train_instances[classs])/self.total_instances)
            weightage=0
            class_words=self.matrix[classs]
            for words in word_counter:
                if(words in class_words.keys()):
                    weightage=class_words[words]+1
                else:
                    weightage=1
                weightage=float(weightage)/(self.unique_words+self.class_words[classs])
                probability+=word_counter[words]*math.log(weightage)
            if(predicted_class==""):
                predicted_value=probability
                predicted_class=classs
            elif(probability>predicted_value):
                predicted_value=probability
                predicted_class=classs
        if predicted_class!=self.current_class:
            self.errors+=1    
            
    
    def Construct_Matrix(self,word_list):
        for word in word_list:
            if word.lower() in self.matrix[self.current_class].keys():
                self.matrix[self.current_class][word.lower()]+=1
            else:
                self.matrix[self.current_class][word.lower()]=1
            if word.lower() not in self.all_word:
                self.all_word.append(word.lower())
        
if __name__=='__main__':
     try:
        training_data_location=sys.argv[1]
        test_data_location=sys.argv[2]
     except:
        print ""
        print "Usage: MNB.py <training data location> <test data location>"
        exit()
     Classifier=NB(training_data_location,test_data_location)
     Classifier.Create_Model()
     Accuracy=Classifier.Test_Data() 
     print "Accuracy of model on test data = ",Accuracy,"%"