-----------------------------------------------------------------------------------------------------------------------------------------------------------

Preprocessing.py:

Usage: preprocessing.py <input location> <output location>



ann.py:

Usage: ann.py <input location> <train-percent-of-data> <number of iterations> <number of hidden layers> <nodes in hidden layer1> <nodes in hidden layer2>...


Pre-requisite: Pandas library

-------------------------------------------------------------------------------------------------------------------------------------------------------------

Aim:-Preprocess input data and construct artificial neural network using back propogation algorithm using sigmoid as activation function.

Assumptions
a.	Pre-processing Unit:
1.	Feature/Attribute Names will not be present in the input file.

b.	ANN Unit:
1.	Class range can be discreate or continuous. 
2.	If data contains only discreate numeric values, it is considered as a regression problem.
Implementation Details:
a.	Pre-processing Unit:
1.	For Categorical values as attribute values in input file, It is converted to numerical value by using the index of the unique list of categorical values. Each value is encoded as index/100+0.4.
2.	Classes having a continuous data range, are normalized using min-max notation to restrict the range to [0,1].
3.	The preprocessor automatically generates random attribute names using a five letter attribute name strategy.

b.	ANN Unit:
1.	For a classification problem the number of output layer neurons is equal to the number of unique classes in the data and each neuron indicates a class.
2.	For regression problem there is only one neuron in output layer.
3.	Learning rate is set at 0.25.
4.	Every Layer is defined as an object having the following properties:
•	Values: list of values of X- neuron in the layer.
Eg values=[1,3,3,1] implies first neuron has 1 as input value, second neuron has 3 etc. 
•	Weights: Is a dictionary of lists where key represent the individual neuron and value represents corresponding weights to next layer.
Eg { 0: [0.34,0.6], 1:[0.3,.98],2:[.65,.54],3: [.67,.95]} 
•	Delta: Delta is a list of delta calculated during the backward pass.
•	The last value in every layer except the output layer is 1 which corresponds to the bias term w0 and it has corresponding weights to next layer.
5.	All layer object is pushed into an array and is iterated for each forward and backward pass.
Experiments Conducted during Development:
1.	Post development to ensure learning and accuracy, an output printing function was used to print the target and the output. This exercise ensured that the with each update the target was converging to output.
2.	During development simple sample datasets were used to develop before validating with bigger dataset. This experiment helped in cornering a lot of potential bugs and eliminating them.
3.	Both regression and classification experiments were conducted on pre-processing.py and ann.py to ensure functionality.
