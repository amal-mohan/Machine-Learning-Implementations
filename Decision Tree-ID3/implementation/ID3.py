import pandas as panda
import numpy as np
import sys
from random import randint
import copy
from random import choice

count =0
leaf_count=0

class Node:
    def    __init__(self,attribute):
        self.attribute=attribute
        self.left_node=None
        self.right_node=None
        self.index="null"
        self.alternate="null"
    def update_left_node(self,n):
        self.left_node=n
    
    def update_right_node(self,n):
        self.right_node=n

def train(training_data):
    processed_training_data=modify_data(training_data)
    #data is converted to the form {attribute1:[value1,value2,...],attribute2:[value1,value2...]...class:[label1,label2...]}
    processed_training_data.dropna(inplace=True)
    processed_training_data.reset_index(drop=True,inplace=True)
    N=trainer(processed_training_data,"null",'o')
    print "----------------Decision Tree-----------------------"
    print_node(N,0,'ro')
    print "----------------------------------------------------"
    return N
    
def print_node(N,depth,direction):
    if(N.attribute=='one'):
        print " : 1"
        return
    elif(N.attribute=='zero'):
        print " : 0"
        return
    else:
        print ""
    if(N.right_node!=None):
        for i in range(0,depth):
            sys.stdout.write('| ')
        sys.stdout.write(N.attribute)
        sys.stdout.write(" = 1")
        print_node(N.right_node,depth+1,'r')
    if(N.left_node!=None):
        for i in range(0,depth):
            sys.stdout.write('| ')
        sys.stdout.write(N.attribute)
        sys.stdout.write(" = 0")
        print_node(N.left_node,depth+1,'l')
        
        
    
        
def trainer(data,parent_object,location):
    #right movement is one and left is zero
    classifying_attribute=attribute_with_maximum_information_gain(data)
    if classifying_attribute=='null':
        return
    elif classifying_attribute=="zero":
        N=Node(classifying_attribute)
        if(location=='l'):
            parent_object.update_left_node(N)
        elif(location=='r'):
            parent_object.update_right_node(N)
        return
    elif classifying_attribute=="one":
        N=Node(classifying_attribute)
        if(location=='l'):
            parent_object.update_left_node(N)
        elif(location=='r'):
            parent_object.update_right_node(N)
        return
    N=Node(classifying_attribute)
    dfa=data.copy()
    dfb=data.copy()
    dfa.set_index([classifying_attribute],inplace=True)
    dfb.set_index([classifying_attribute],inplace=True)
    one=dfa.drop([0])
    zero=dfb.drop([1])
    one=one.reset_index()
    zero=zero.reset_index()

    left_majority=Majority(classifying_attribute,zero)
    right_majority=Majority(classifying_attribute,one)
    A=Node(left_majority)
    B=Node(right_majority)
    N.update_left_node(A)
    N.update_right_node(B)
    M=Majority(classifying_attribute,data)
    M1=Node(M)
    N.alternate=M1
    if parent_object!="null" and location=='l':
        parent_object.update_left_node(N)
    elif parent_object!="null" and location=='r':
        parent_object.update_right_node(N)
    right_tree=classify_data(classifying_attribute,data,'r')
    left_tree=classify_data(classifying_attribute,data,'l')
    trainer(left_tree,N,'l')
    trainer(right_tree,N,'r')
    return N

def Majority(classifying_attribute,data):
    if(len(data)==1):
        if(data.Class[0]==1):
            return "one"
        else:
            return "zero"
    df0=data.copy()
    df1=data.copy()
    one=0
    zero=0
    df0.set_index([classifying_attribute],inplace=True)
    df1.set_index([classifying_attribute],inplace=True)
    one=len(df0.drop([0]))
    zero=len(df1.drop([1]))
    if(zero>one):
        return "zero"
    else:
        return "one"
 
def modify_data(training_data):
    train_data = panda.read_csv(training_data)
    return train_data
    
        
def attribute_with_maximum_information_gain(data):
    max_information_gain=-1
    if(len(data.index)==1):
        if data["Class"][0]==0:
            return "zero"
        else:
            return "one"
    attribute="null"
    parent_entropy=calculate_parent_entropy(data)
    if(parent_entropy==-1):
        return "zero"
    elif(parent_entropy==-2):
        return "one"
    for cols in data:
        if cols !='Class':
            information_gain=calculate_information_gain(data,cols,parent_entropy)
            if(max_information_gain==-1):
                max_information_gain=information_gain
                attribute=cols
            elif(max_information_gain<information_gain):
                max_information_gain=information_gain
                attribute=cols            
    if(max_information_gain==0 or attribute =="null"):
        return choice['zero','one']
    return attribute

def calculate_parent_entropy(data):
    column=data[["Class"]]
    zero=0
    one=0
    for index,rows in column.iterrows():
        if rows['Class']==1:
            one+=1
        elif rows['Class']==0:
            zero+=1
    if(one== 0) :
        return -1
    if(zero==0):
        return -2
    return (-(float(zero)/(one+zero))*np.log2(np.array([float(zero)/(one+zero)]))[0]-((float(one)/(one+zero))*np.log2(np.array([float(one)/(one+zero)]))[0]))
    
def calculate_information_gain(data,cols,parent_entropy):
    column=data[[cols,"Class"]]
    right_zero=0
    right_one=0
    left_zero=0
    left_one=0
    for i in range(0,len(column)):
        row_class= column.Class.loc[i]
        row_col= column[cols].loc[i]
        if(type(row_col)!=panda.core.series.Series):
            if row_col==1 and row_class==1:
                right_one+=1
            elif row_col==1 and row_class==0:
                right_zero+=1
            elif row_col==0 and row_class==1:
                left_one+=1
            elif row_col==0 and row_class==0:
                left_zero+=1
    left=left_one+left_zero
    right=right_one+right_zero
    total=left+right
    if(right_one==0 or right_zero==0):
        right_entropy=0
    else:
        right_entropy=(-(float(right_zero)/(right_one+right_zero))*np.log2(np.array([float(right_zero)/(right_one+right_zero)]))[0]-((float(right_one)/(right_one+right_zero))*np.log2(np.array([float(right_one)/(right_one+right_zero)]))[0]))
    if(left_one==0 or left_zero==0):
        left_entropy=0
    else:
        left_entropy=(-(float(left_zero)/(left_one+left_zero))*np.log2(np.array([float(left_zero)/(left_one+left_zero)]))[0]-((float(left_one)/(left_one+left_zero))*np.log2(np.array([float(left_one)/(left_one+left_zero)]))[0]))
    information_gain=parent_entropy-((float(right)/total)*right_entropy+(float(left)/total)*left_entropy)
    return information_gain

def classify_data(classifying_attribute,data,direction):
    df=data.copy()
    if(len(df.index)==1):
        return
    if(direction=='r'):
        df.set_index([classifying_attribute],inplace=True)
        df=df.drop([0])
    elif(direction=='l'):
        df.set_index([classifying_attribute],inplace=True)
        df=df.drop([1])
    df=df.reset_index(drop=True)
    return df.copy()

def number_nodes(N,h):
    for i in range(1, h+1):
        number_node(N, i)

def number_node(N , height):
    global count
    global leaf_count
    if N ==None:
        return
    if height == 1:
        if N.attribute!="one" and N.attribute!="zero" :
            N.index=count+1
            count+=1
        else:
            leaf_count+=1     
    elif height > 1 :
        number_node(N.left_node , height-1)
        number_node(N.right_node , height-1)


def height_tree(N):
    if N != None:
        left_height = height_tree(N.left_node)
        right_height = height_tree(N.right_node)
        if left_height > right_height :
            return left_height+1
        else:
            return right_height+1
    else:
        return 0

def copy_tree(N):
    if N ==None:
        return
    P=copy.copy(N)
    P.left_node=copy_tree(P.left_node)
    P.right_node=copy_tree(P.right_node)
    return P

def count_nodes(N):
    if N ==None:
        return
    
    P=copy.copy(N)
    P.left_node=copy_tree(P.left_node)
    P.right_node=copy_tree(P.right_node)
    return P

def prune_tree(pruning_factor,N,pre_prune_accuracy,data):
    nodes_to_prune=int(pruning_factor*count)
    a=int(nodes_to_prune*1.5)
    j=[]
    for i in range(0,nodes_to_prune):
        while(1):
            node_index=randint(count-a,count)
            while(node_index in j):
                node_index=randint(count-a,count)
            j.append(node_index)
            A=get_node(N,N,node_index,"ro")
            if A!=None:
                if A[2]=='ro':
                    A[2]=A[2].alternate
                if A[2]=='l':
                    A[0].left_node=A[1].alternate
                if A[2]=='r':
                    A[0].right_node=A[1].alternate
            pruned_accuracy=validate_data(data,N)    
            if(pre_prune_accuracy>pruned_accuracy):
                if A[2]=='l':
                    A[0].left_node=A[1]
                if A[2]=='r':
                    A[0].right_node=A[1]
            else:
                break
        
def get_node(P,N,index,direction):
    if N ==None:
        return
    if N.index==index:
        return [P,N,direction]
    A=get_node(N,N.left_node,index,"l")
    if A!=None:
        return A
    A=get_node(N,N.right_node,index,"r")
    if A!=None:
        return A
        
def validate_data(data,N):
    df=panda.read_csv(data)
    correct_nodes=0
    df.dropna(inplace=True)
    df.reset_index(drop=True,inplace=True)
    total_nodes=len(df)
    for index,rows in df.iterrows():
        #print index
        N1=N
        while(N1.attribute!="zero" and N1.attribute!="one"):
            if df[N1.attribute][index]==1:
                N1=N1.right_node
            elif df[N1.attribute][index]==0:
                N1=N1.left_node
        if (N1.attribute=="zero" and df['Class'][index]==0) or (N1.attribute=="one" and df['Class'][index]==1):
            correct_nodes+=1	
    return float(correct_nodes)/total_nodes
    
if __name__=="__main__":
    try:
        training_set=sys.argv[1]
        validation_set=sys.argv[2]
        test_set=sys.argv[3]
        pruning_factor=float(sys.argv[4])
    except:
        print ""
        print "Usage: ID3.py <training_data location> <validation data location> <test data location> <pruning factor>"
        exit()
        
    N=train(training_set)
    h = height_tree(N)
    number_nodes(N,h)
    pruned_accuracy=0
    print "Pre-Pruned Accuracy -------------------------------------"
    print "Number of training instances = ",len(panda.read_csv(training_set).dropna())
    print "Number of training attributes = ",len(panda.read_csv(training_set).columns)-1
    print "Total number of nodes in the tree = ",leaf_count+count
    print "Number of leaf nodes in the tree = ",leaf_count
    orginal_accuracy=validate_data(training_set,N)
    print "Accuracy of the model on the training dataset = ",orginal_accuracy*100,"%"
    print ""
    print "Number of validation instances = ",len(panda.read_csv(validation_set).dropna())
    print "Number of validation attributes = ",len(panda.read_csv(validation_set).columns)-1
    orginal_validation_accuracy=validate_data(validation_set,N)
    print "Accuracy of the model on the validation dataset before pruning = ",orginal_validation_accuracy*100,"%" 
    print ""
    print "Number of testing instances = ",len(panda.read_csv(test_set).dropna())
    print "Number of testing attributes = ",len(panda.read_csv(test_set).columns)-1
    orginal_test_accuracy=validate_data(test_set,N)
    print "Accuracy of the model on the testing dataset = ",orginal_test_accuracy*100,"%"
    P=copy_tree(N)
    prune_tree(pruning_factor,P,orginal_validation_accuracy,validation_set)
    pruned_accuracy=validate_data(validation_set,P)
    pruned_training_accuracy=validate_data(training_set,P)
    print ""
    print "Post-Pruned Accuracy ------------------------------------- "
    print "Number of training instances = ",len(panda.read_csv(training_set).dropna())
    print "Number of training attributes = ",len(panda.read_csv(training_set).columns)-1
    count =0
    leaf_count=0
    number_nodes(P,h)
    print "Total number of nodes in the tree = ",leaf_count+count
    print "Number of leaf nodes in the tree = ",leaf_count
    print "Accuracy of the model on the training dataset = ",pruned_training_accuracy*100,"%"
    print ""
    print "Number of validation instances = ",len(panda.read_csv(validation_set).dropna())
    print "Number of validation attributes = ",len(panda.read_csv(validation_set).columns)-1
    print "Accuracy of the model on the validation dataset after pruning = ",pruned_accuracy*100,"%" 
    print ""
    print "Number of testing instances = ",len(panda.read_csv(test_set).dropna())
    print "Number of testing attributes = ",len(panda.read_csv(test_set).columns)-1
    pruned_test_accuracy=validate_data(test_set,P)
    print "Accuracy of the model on the testing dataset = ",pruned_test_accuracy*100,"%"
    
    
    
 
  
 
  
    
    
