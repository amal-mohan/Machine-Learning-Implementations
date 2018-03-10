# -*- coding: utf-8 -*-
"""
Created on Tue Oct 03 10:43:38 2017

"""

import sys
import pandas as panda
import numpy as np
import string as s
import random as r

class Preprocessing:
    
    def __init__(self,data_path):
        #initialize dataset to dataframe
        #drop empty values
        number_of_features=len(panda.read_csv(data_path).columns)
        col=self.Number_Features(number_of_features)
        self.data_frame=panda.read_csv(data_path,names=col)
        self.data_frame.dropna(inplace=True)
        self.data_frame.reset_index(drop=True,inplace=True)
        self.Class=panda.DataFrame(self.data_frame.iloc[:,-1])
        self.Features=self.data_frame.drop(list(self.data_frame)[-1],axis=1)
    
    def Number_Features(self,number_of_features):
        #random naming each attribute of the data
        f_name=[]
        f_name.append("")
        for i in range(0,number_of_features-1):
            k=""
            while(k in f_name):
                k=""
                for i in range(5):
                    li=r.choice(s.letters)
                    k=k+li
            f_name.append(k)
        f_name.remove("")
        f_name.append("Class")
        return f_name
        
    def Scale(self):
        #parent function to scale data
        for column in self.Features:
            if self.Features[column].dtype=='int64' or self.Features[column].dtype=='float64':
                self.Numeric_Standadizer(column)
            else:
                self.Categorical_Standadizer(column)
        for column in self.Class:
            if self.Class[column].dtype=='int64' or self.Class[column].dtype=='float64':
                self.Numeric_Standadizer_Class(column)
        
    def Numeric_Standadizer_Class(self,col_id):
        #standadizes continuous data in class
        minimum=self.Class[col_id].min()
        maximum=self.Class[col_id].max()
        for i in range(0,len(self.Class[col_id])):
            self.Class.loc[i,col_id]=float(self.Class.loc[i,col_id]-minimum)/(maximum-minimum)
    
    def Numeric_Standadizer(self,col_id):
        #standadizes numeric attributes
        mean=self.Features[col_id].mean()
        standard_deviation=self.Features[col_id].std()
        for i in range(0,len(self.Features[col_id])):
            self.Features.loc[i,col_id]=float(self.Features.loc[i,col_id]-mean)/standard_deviation
        
        
    def Categorical_Standadizer(self,col_id):
        #standardizes categorical data
        values= self.Features[col_id].unique()
        for i in range(0,len(self.Features[col_id])):
            self.Features.loc[i,col_id]=float(np.where(values==self.Features.loc[i,col_id])[0].tolist()[0])/100+0.4
         
        
    def Save_Data(self,data_output_location):
        #saves data to destination
        output_frame=panda.concat([self.Features, self.Class], axis=1)
        output_frame.to_csv(data_output_location,index=False)

if __name__=='__main__':
     try:
        data_input_location=sys.argv[1]
        data_output_location=sys.argv[2]
     except:
        print ""
        print "Usage: preprocessing.py <input location> <output location>"
        exit()
     Preprocessor=Preprocessing(data_input_location)
     Preprocessor.Scale()
     Preprocessor.Save_Data(data_output_location)
     
        
    