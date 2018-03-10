# -*- coding: utf-8 -*-
"""
Created on Fri Oct 06 00:37:53 2017

"""
import sys
import pandas as panda
import random
import math


class Layer:
    
    def __init__(self,layer_size,next_layer_size=0,bias_required=1):
        #bias required is 0 for output layer and 1 for other layers denotes input bias required
        self.delta=[]
        self.values=[]
        self.weights={}
        for i in range(0,layer_size):
            self.values.append(0)
        if bias_required==1:
            self.values.append(1)
            self.weights={}
            for i in range(0,layer_size+bias_required):
                self.weights[i]=[]
                for j in range(0,next_layer_size):
                    w=float(random.randint(0,101))/100
                    self.weights[i].append(w)

    def update_input_layer(self,data):
        self.values= list(list(data)[1])
        self.values.pop()
        self.values.append(1)
        
    def get_value(self):
        return self.values
    
    def get_weights(self):
        return self.weights
    
    def update_value(self,value_id,value):
        self.values[value_id]=value
    
    def update_delta(self,delta):
        self.delta=delta
    
    def get_delta(self):
        return self.delta
    
    def update_weight(self,weights):
        self.weights=weights
        
    
class ANN:
     #attributes
#     self.data_frame
#     self.data_length
#     self.test_data
#     self.train_count
#     self.train_data
#     self.train_index
#     self.classes     
#     self.layer_sizes
#     
    def __init__(self,data_path,train_percent,total_hidden_layer,layer_sizes):
        self.data_frame=panda.read_csv(data_path)
        self.data_length=len(self.data_frame)
        self.train_count= int(self.data_length*train_percent/100)
        self.train_data=panda.DataFrame(self.data_frame)
        self.test_data=panda.DataFrame(self.data_frame)
        self.find_problem()
        self.layer_sizes=layer_sizes
        self.append_input_output_layer()
        self.layers=[]
        self.eta=0.25
                
    def create_neural_network(self,iterations):
        for i in range(0,len(self.layer_sizes)):
            if(i==len(self.layer_sizes)-1):
                self.layers.append(Layer(self.layer_sizes[i],bias_required=0))
            else:
                self.layers.append(Layer(self.layer_sizes[i],self.layer_sizes[i+1],bias_required=1))        
        for i in range(0,iterations):
            for row in self.train_data.iterrows():
                target=self.calculate_target(row)
               
                for j in range(0,len(self.layer_sizes)):
                    if(j==0):
                        self.layers[0].update_input_layer(row)
                    else:
                        self.forward_pass(j)
                for j in xrange(len(self.layer_sizes)-1,0,-1):
                    if(j==len(self.layer_sizes)-1):
                        self.backward_pass(j,1,target)
                    else:
                        self.backward_pass(j,0,target)
                for j in range(0,len(self.layer_sizes)-1):
                    self.update_weights(j)
            error=self.test_neural_network('train')
            if(error*1000==0):
                break                             
    
    def update_weights(self,layer_id):
        layer=self.layers[layer_id]
        next_layer=self.layers[layer_id+1]
        next_layer_delta=next_layer.get_delta()
        neurons=self.layer_sizes[layer_id]
        layer_values=layer.get_value()
        layer_weights=layer.get_weights()
        total_weight=len(layer_weights[0])
        for neuron in range(0,neurons+1):
            for weight in range(0,total_weight):
                layer_weights[neuron][weight]+=self.eta*next_layer_delta[weight]*layer_values[neuron]
        layer.update_weight(layer_weights)                                    
    
    def calculate_target(self,row):
        if self.problem=='regression':
            target=[]
            target.append(list(list(row)[1]).pop())
            return target
        else:
            target=[]
            total_classes=len(self.classes)
            a=str(list(list(row)[1]).pop())
            try:
                index=self.classes.index(a)
            except:
                index=-1
            for i in range(0,total_classes):
                if(i==index):
                    target.append(1)
                else:
                    target.append(0)
            return target
    
    def forward_pass(self,layer_number):
        neurons=self.layer_sizes[layer_number]
        layer=self.layers[layer_number]
        for i in range(0,neurons):
            net=self.calculate_net(i,self.layers[layer_number-1])    
            output=self.sigmoid(net)
            layer.update_value(i,output)
#        if(self.problem=='classification' and layer_number==len(self.layer_sizes)-1):
#            output_layer=self.layers[len(self.layers)-1].get_value()
#            if(len(output_layer)>1):
#                max_val=output_layer[0]
#                max_index=0
#                for i in range(1,len(output_layer)):
#                    if(output_layer[i]>max_val):
#                        max_val=output_layer[i]
#                        max_index=i
#                for i in range(0,len(output_layer)):
#                    if(i==max_index):
#                        layer.update_value(i,1)
#                    else:
#                        layer.update_value(i,0)
                        

        
    def sigmoid(self,net):
        return 1/(1+math.exp(-(net)))
        
    def calculate_net(self,neuron_id,previous_layer):
        netj=0
        input_layer=previous_layer.get_value()
        weights=previous_layer.get_weights()
        for i in range(0,len(input_layer)):
            netj+=input_layer[i]*weights[i][neuron_id]
        return netj
    
    def backward_pass(self,layer_number,output,target):      
        if(output==1):
            neurons=self.layer_sizes[layer_number]
            layer=self.layers[layer_number]
            layer_values=layer.get_value()
            delta=[]
            for neuron in range(0,neurons):
                delta.append((target[neuron]-layer_values[neuron])*layer_values[neuron]*(1-layer_values[neuron]))
            layer.update_delta(delta)    
        else:
            neurons=self.layer_sizes[layer_number]
            layer=self.layers[layer_number]
            layer_values=layer.get_value()
            layer_weights=layer.get_weights()
            next_layer=self.layers[layer_number+1]
            next_delta=next_layer.get_delta()
            delta=[]
            for neuron in range(0,neurons):
                sum_weight_delta=0
                for i in range(0,len(next_delta)):
                    sum_weight_delta+=next_delta[i]*layer_weights[neuron][i]
                delta.append(sum_weight_delta*layer_values[neuron]*(1-layer_values[neuron]))            
            layer.update_delta(delta)
            
    def append_input_output_layer(self):
        self.layer_sizes.insert(0,len(self.train_data.columns)-1)
        if(self.problem=='regression'):
            self.layer_sizes.append(1)
        else:
            self.classes=list(self.train_data.iloc[:,-1].unique())
            self.layer_sizes.append(len(self.classes))
    
    def find_problem(self):
        if self.data_frame['Class'].dtype=='int64' or self.data_frame['Class'].dtype=='float64':  
            self.problem='regression'
        else:
            self.problem='classification'
        
    def split_data(self):
        self.train_index=[]
        for i in range(0,self.train_count):
            index=random.randint(0,self.train_count)
            while(index in self.train_index):
                index=random.randint(0,self.train_count)
            self.train_index.append(index)
        for i in xrange(self.data_length-1,-1,-1):
            if(i not in self.train_index):
                self.train_data=self.train_data.drop(self.train_data.index[i])
        for i in xrange(self.data_length-1,-1,-1):
            if(i in self.train_index):
                self.test_data=self.test_data.drop(self.test_data.index[i])
    
    def test_neural_network(self,data_type):
        if data_type=='train':
            data=self.train_data
        else:
            data=self.test_data
        error=0
        for row in data.iterrows():
            target=self.calculate_target(row)
            for j in range(0,len(self.layer_sizes)):
               if(j==0):
                    self.layers[0].update_input_layer(row)
               else:
                    self.forward_pass(j)         
            error+=self.mean_square_error(target)
        average_error=float(error)/(len(data)*2*len(target))
        return average_error
        
    def mean_square_error(self,target):
       output_layer=self.layers[len(self.layers)-1].get_value()
       error=0
#       if(self.problem=='classification'):
#            if(len(output_layer)>1):
#                max_val=output_layer[0]
#                max_index=0
#                for i in range(1,len(output_layer)):
#                    if(output_layer[i]>max_val):
#                        max_val=output_layer[i]
#                        max_index=i
#                op=[]
#                for i in range(0,len(output_layer)):
#                    if(i==max_index):
#                        op.append(1)
#                    else:
#                        op.append(0)
#            output_layer=op
#       print output_layer, " :",target
        
       for i in range(0,len(target)):
          error+=math.pow(target[i]-output_layer[i],2)
       #error=float(error)/(2*len(target))
       return error
    
    def print_neural_network(self):
        for i in range(0,len(self.layer_sizes)-1):
            layer=self.layers[i]
            if(i==0):
                print "Layer 0 (Input Layer):"
            elif(i==len(self.layer_sizes)-2):
                print "Layer n (Last Hidden Layer):"
            else:
                print "Layer ",i," (",i," Hidden Layer)"
            weight=layer.get_weights()
            for j in range(0,len(layer.get_value())):
                if(j==len(layer.get_value())-1):
                    print "        Bias Weights: ",weight[j]
                else:
                    print "        Neuron ",j+1," weights: ",weight[j]

if __name__=='__main__':
    try:
        data_input_location=sys.argv[1]
        train_percent=sys.argv[2]
        number_iterations=sys.argv[3]
        total_hidden_layer=int(sys.argv[4])
        layer_sizes=[]
        for i in range(1,total_hidden_layer+1):
            layer_sizes.append(int(sys.argv[4+i]))
    except:
        print ""
        print "Usage: ann.py <input location> <train percent> <number of iterations> <number of hidden layers> <nodes in hidden layer1> <nodes in hidden layer2>..."
        exit()
    A=ANN(data_input_location,float(train_percent),total_hidden_layer,layer_sizes)
    A.split_data()
    A.create_neural_network(int(number_iterations))
    A.print_neural_network()
    train_error=A.test_neural_network('train')
    test_error=A.test_neural_network('test')
    print "Mean Square Error"
    print "Total training error: ", train_error
    print "Total test error: ",test_error