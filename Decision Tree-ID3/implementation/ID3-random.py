import pandas as panda
import numpy as np
import sys
from random import randint,choice

import copy

count =0
leaf_count=0
depth=0
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

def train_random(training_data):
    processed_training_data=modify_data(training_data)
    #data is converted to the form {attribute1:[value1,value2,...],attribute2:[value1,value2...]...class:[label1,label2...]}
    processed_training_data.dropna(inplace=True)
    processed_training_data.reset_index(drop=True,inplace=True)
    cols=processed_training_data.columns
    cols=list(cols)
    cols.remove("Class")
    a=randint(len(cols)/2,len(cols))
    N=trainer_random(cols,"null",'o',a,0)
    
    return N

        
def trainer_random(data,parent_object,location,a,n):
    #right movement is one and left is zero
    d=copy.copy(data)
    if(len(d)==0 or a ==n):
        ch=choice(['zero','one'])
        N=Node(ch)
        if parent_object!="null" and location=='l':
            parent_object.update_left_node(N)
        elif parent_object!="null" and location=='r':
            parent_object.update_right_node(N)    
        return N
    classifying_attribute=random_attribute(d)
    if(classifying_attribute=='zero'or classifying_attribute=='one'):
            classifying_attribute=random_attribute(d)
    if(n>=5):
        d.append(choice(['zero','one']))
        d.append(choice(['zero','one']))
        #data.append('one')
    d.remove(classifying_attribute)
    d.remove(random_attribute(d))
    if classifying_attribute=="zero":
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
    if parent_object!="null" and location=='l':
        parent_object.update_left_node(N)
    elif parent_object!="null" and location=='r':
        parent_object.update_right_node(N)    
    trainer_random(d,N,'l',a,n+1)
    trainer_random(d,N,'r',a,n+1)
    return N

def random_attribute(data):
    return choice(data)

def train(training_data):
    processed_training_data=modify_data(training_data)
    processed_training_data.dropna(inplace=True)
    processed_training_data.reset_index(drop=True,inplace=True)
    N=trainer(processed_training_data,"null",'o')
    print "----------------ID3 Decision Tree-------------- -----"
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
    else:
        print "None"
        
    
        
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
    # returns attribute with maximum information gain
    # returns null if no attribute is present or information gain is 0
    max_information_gain=-1
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
    if(max_information_gain==0):
        return "null"
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
    global depth
    depth=0
    for i in range(1, h+1):
        number_node(N, i,0)
    

def number_node(N , height,c):
    global count
    global leaf_count
    global depth
    if N ==None:
        return
    if height == 1:
        if N.attribute!="one" and N.attribute!="zero" :
            N.index=count+1
            count+=1
        else:
            leaf_count+=1
            depth+=c
    elif height > 1 :
        number_node(N.left_node , height-1,c+1)
        number_node(N.right_node , height-1,c+1)
        


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
    except:
       print ""
       print "Usage: ID3.py <training_data location> <validation data location> <test data location> <pruning factor>"
       exit()
    N=train(training_set)
    N1=train_random(training_set)
    print "------------Random attribute selection Decision Tree---"
    print_node(N1,0,'ro')
    print "-------------------------------------------------------"
    
    h = height_tree(N)
    number_nodes(N,h)
    average_depth_ID3=float(depth)/leaf_count
    print "                                                      Average Depth   Number of nodes"
    print "Tree constructed through ID3                        ",average_depth_ID3,"   ",leaf_count+count
    count =0
    leaf_count=0
    N1=train_random(training_set)
    number_nodes(N1,h)
    average_depth_ran=float(depth)/leaf_count
    print "Tree constructed through random attribute selection ",average_depth_ran,"   ",leaf_count+count
    print "Random selection algorithm accuracy  ------------------------------------- "
    print "Run   Accuracy of tree constructed using random attribute selection"
    for i in range(0,5):
        sys.stdout.write(str(i+1))
        sys.stdout.write("    ")
        accuracy=validate_data(test_set,N1)
        sys.stdout.write(str(accuracy*100)+"%")
        print ""
        N1=train_random(training_set)
        count =0
        leaf_count=0
        number_nodes(N1,h)
    count =0
    leaf_count=0
    N1=train_random(training_set)
    number_nodes(N,h)
    accuracy=validate_data(test_set,N)
    print "accuracy of ID3 decision tree= ",accuracy*100,"%"
    