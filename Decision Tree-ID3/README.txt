---------------------------------------------------------------------------------------------------------------------------------------
Language Used:  Python

Usage: ID3.py <training_data location> <validation data location> <test data location> <pruning factor>

Usage: ID3-random.py <training_data location> <validation data location> <test data location> <pruning factor>

Pre-requisite: Pandas library

---------------------------------------------------------------------------------------------------------------------------------------

ID3.py

Aim-Construct decision tree-binary classifier for binary attributes using ID3 algorithm. 

Assumptions:
1.	The tree is traversed to left when zero is input and to right when one is input.
2.	The first attribute found is used for classification when more than one attribute has same information gain.
3.	If there is only an instance of data left, the class of instance is value for the node in tree.

Best Result:
----------------Decision Tree-----------------------
XO = 1
| XI = 1
| | XT = 1
| | | XS = 1
| | | | XL = 1
| | | | | XH = 1 : 0
| | | | | XH = 0
| | | | | | XD = 1
| | | | | | | XB = 1 : 0
| | | | | | | XB = 0 : 1
| | | | | | XD = 0
| | | | | | | XQ = 1
| | | | | | | | XB = 1 : 1
| | | | | | | | XB = 0 : 0
| | | | | | | XQ = 0 : 0
| | | | XL = 0
| | | | | XD = 1
| | | | | | XG = 1 : 1
| | | | | | XG = 0 : 0
| | | | | XD = 0
| | | | | | XU = 1
| | | | | | | XB = 1
| | | | | | | | XG = 1 : 0
| | | | | | | | XG = 0
| | | | | | | | | XH = 1 : 1
| | | | | | | | | XH = 0 : 0
| | | | | | | XB = 0
| | | | | | | | XE = 1
| | | | | | | | | XC = 1 : 0
| | | | | | | | | XC = 0 : 1
| | | | | | | | XE = 0 : 1
| | | | | | XU = 0 : 1
| | | XS = 0
| | | | XQ = 1
| | | | | XM = 1 : 0
| | | | | XM = 0
| | | | | | XN = 1
| | | | | | | XP = 1 : 1
| | | | | | | XP = 0
| | | | | | | | XB = 1 : 1
| | | | | | | | XB = 0
| | | | | | | | | XF = 1 : 1
| | | | | | | | | XF = 0 : 0
| | | | | | XN = 0
| | | | | | | XU = 1 : 0
| | | | | | | XU = 0 : 1
| | | | XQ = 0
| | | | | XK = 1
| | | | | | XD = 1 : 0
| | | | | | XD = 0
| | | | | | | XF = 1 : 1
| | | | | | | XF = 0 : 0
| | | | | XK = 0
| | | | | | XC = 1 : 1
| | | | | | XC = 0
| | | | | | | XR = 1
| | | | | | | | XB = 1 : 0
| | | | | | | | XB = 0
| | | | | | | | | XD = 1 : 1
| | | | | | | | | XD = 0 : 0
| | | | | | | XR = 0
| | | | | | | | XH = 1 : 1
| | | | | | | | XH = 0
| | | | | | | | | XE = 1 : 1
| | | | | | | | | XE = 0 : 0
| | XT = 0
| | | XH = 1
| | | | XJ = 1
| | | | | XS = 1
| | | | | | XG = 1
| | | | | | | XC = 1
| | | | | | | | XD = 1 : 0
| | | | | | | | XD = 0 : 1
| | | | | | | XC = 0 : 1
| | | | | | XG = 0
| | | | | | | XB = 1
| | | | | | | | XD = 1
| | | | | | | | | XE = 1 : 1
| | | | | | | | | XE = 0 : 0
| | | | | | | | XD = 0 : 1
| | | | | | | XB = 0 : 0
| | | | | XS = 0 : 1
| | | | XJ = 0
| | | | | XC = 1
| | | | | | XM = 1
| | | | | | | XF = 1 : 1
| | | | | | | XF = 0
| | | | | | | | XR = 1 : 0
| | | | | | | | XR = 0 : 1
| | | | | | XM = 0 : 0
| | | | | XC = 0
| | | | | | XN = 1
| | | | | | | XF = 1 : 0
| | | | | | | XF = 0
| | | | | | | | XG = 1 : 0
| | | | | | | | XG = 0 : 1
| | | | | | XN = 0 : 1
| | | XH = 0
| | | | XP = 1
| | | | | XS = 1 : 1
| | | | | XS = 0
| | | | | | XD = 1
| | | | | | | XM = 1 : 1
| | | | | | | XM = 0
| | | | | | | | XC = 1 : 0
| | | | | | | | XC = 0 : 1
| | | | | | XD = 0
| | | | | | | XC = 1 : 0
| | | | | | | XC = 0
| | | | | | | | XJ = 1
| | | | | | | | | XB = 1
| | | | | | | | | | XG = 1 : 0
| | | | | | | | | | XG = 0 : 1
| | | | | | | | | XB = 0 : 0
| | | | | | | | XJ = 0
| | | | | | | | | XN = 1 : 1
| | | | | | | | | XN = 0 : 0
| | | | XP = 0
| | | | | XF = 1
| | | | | | XQ = 1
| | | | | | | XK = 1 : 1
| | | | | | | XK = 0 : 0
| | | | | | XQ = 0
| | | | | | | XK = 1
| | | | | | | | XC = 1 : 1
| | | | | | | | XC = 0 : 0
| | | | | | | XK = 0 : 1
| | | | | XF = 0 : 0
| XI = 0
| | XM = 1
| | | XQ = 1 : 0
| | | XQ = 0
| | | | XF = 1 : 0
| | | | XF = 0
| | | | | XL = 1
| | | | | | XC = 1 : 1
| | | | | | XC = 0
| | | | | | | XB = 1
| | | | | | | | XP = 1 : 1
| | | | | | | | XP = 0 : 0
| | | | | | | XB = 0 : 0
| | | | | XL = 0
| | | | | | XC = 1
| | | | | | | XH = 1
| | | | | | | | XU = 1 : 1
| | | | | | | | XU = 0
| | | | | | | | | XB = 1 : 0
| | | | | | | | | XB = 0
| | | | | | | | | | XD = 1 : 0
| | | | | | | | | | XD = 0 : 1
| | | | | | | XH = 0 : 1
| | | | | | XC = 0 : 1
| | XM = 0
| | | XQ = 1
| | | | XJ = 1
| | | | | XL = 1 : 1
| | | | | XL = 0
| | | | | | XH = 1
| | | | | | | XK = 1 : 1
| | | | | | | XK = 0
| | | | | | | | XU = 1 : 0
| | | | | | | | XU = 0 : 1
| | | | | | XH = 0 : 0
| | | | XJ = 0
| | | | | XN = 1 : 0
| | | | | XN = 0
| | | | | | XP = 1
| | | | | | | XB = 1 : 0
| | | | | | | XB = 0
| | | | | | | | XF = 1 : 1
| | | | | | | | XF = 0 : 0
| | | | | | XP = 0 : 1
| | | XQ = 0
| | | | XF = 1 : 0
| | | | XF = 0
| | | | | XH = 1 : 1
| | | | | XH = 0
| | | | | | XB = 1
| | | | | | | XC = 1 : 0
| | | | | | | XC = 0 : 1
| | | | | | XB = 0 : 0
XO = 0
| XM = 1
| | XB = 1
| | | XI = 1
| | | | XC = 1 : 0
| | | | XC = 0
| | | | | XK = 1 : 0
| | | | | XK = 0
| | | | | | XP = 1
| | | | | | | XS = 1 : 0
| | | | | | | XS = 0
| | | | | | | | XG = 1
| | | | | | | | | XF = 1 : 1
| | | | | | | | | XF = 0 : 0
| | | | | | | | XG = 0 : 1
| | | | | | XP = 0 : 1
| | | XI = 0 : 0
| | XB = 0
| | | XD = 1
| | | | XC = 1 : 0
| | | | XC = 0
| | | | | XF = 1
| | | | | | XJ = 1
| | | | | | | XE = 1
| | | | | | | | XT = 1 : 1
| | | | | | | | XT = 0
| | | | | | | | | XG = 1 : 0
| | | | | | | | | XG = 0 : 1
| | | | | | | XE = 0
| | | | | | | | XG = 1 : 0
| | | | | | | | XG = 0
| | | | | | | | | XI = 1 : 0
| | | | | | | | | XI = 0 : 1
| | | | | | XJ = 0 : 1
| | | | | XF = 0
| | | | | | XG = 1
| | | | | | | XP = 1 : 0
| | | | | | | XP = 0
| | | | | | | | XS = 1 : 1
| | | | | | | | XS = 0 : 0
| | | | | | XG = 0 : 0
| | | XD = 0
| | | | XG = 1
| | | | | XU = 1
| | | | | | XI = 1 : 1
| | | | | | XI = 0 : 0
| | | | | XU = 0 : 1
| | | | XG = 0
| | | | | XF = 1
| | | | | | XJ = 1
| | | | | | | XC = 1 : 1
| | | | | | | XC = 0
| | | | | | | | XT = 1 : 1
| | | | | | | | XT = 0
| | | | | | | | | XL = 1 : 0
| | | | | | | | | XL = 0
| | | | | | | | | | XE = 1 : 1
| | | | | | | | | | XE = 0
| | | | | | | | | | | XI = 1 : 1
| | | | | | | | | | | XI = 0 : 0
| | | | | | XJ = 0
| | | | | | | XN = 1
| | | | | | | | XE = 1 : 0
| | | | | | | | XE = 0
| | | | | | | | | XK = 1 : 1
| | | | | | | | | XK = 0 : 0
| | | | | | | XN = 0 : 1
| | | | | XF = 0 : 0
| XM = 0
| | XF = 1 : 0
| | XF = 0
| | | XB = 1
| | | | XD = 1
| | | | | XI = 1
| | | | | | XG = 1 : 0
| | | | | | XG = 0 : 1
| | | | | XI = 0 : 0
| | | | XD = 0 : 0
| | | XB = 0
| | | | XG = 1
| | | | | XD = 1
| | | | | | XE = 1
| | | | | | | XK = 1 : 1
| | | | | | | XK = 0 : 0
| | | | | | XE = 0 : 0
| | | | | XD = 0
| | | | | | XS = 1
| | | | | | | XC = 1
| | | | | | | | XH = 1 : 1
| | | | | | | | XH = 0 : 0
| | | | | | | XC = 0 : 1
| | | | | | XS = 0 : 0
| | | | XG = 0 : 0
----------------------------------------------------
Pre-Pruned Accuracy -------------------------------------
Number of training instances =  600
Number of training attributes =  20
Total number of nodes in the tree =  275
Number of leaf nodes in the tree =  138
Accuracy of the model on the training dataset =  100.0 %

Number of validation instances =  2000
Number of validation attributes =  20
Accuracy of the model on the validation dataset before pruning =  75.9 %

Number of testing instances =  2000
Number of testing attributes =  20
Accuracy of the model on the testing dataset =  75.85 %

Post-Pruned Accuracy ------------------------------------- 
Number of training instances =  600
Number of training attributes =  20
Total number of nodes in the tree =  175
Number of leaf nodes in the tree =  88
Accuracy of the model on the training dataset =  91.0 %

Number of validation instances =  2000
Number of validation attributes =  20
Accuracy of the model on the validation dataset after pruning =  76.85 %

Number of testing instances =  2000
Number of testing attributes =  20
Accuracy of the model on the testing dataset =  77.05 %

-----------------------------------------------------------------------------------------------------------------------------------------------------------
ID3-random.py

Aim:- Compare ID3 decision tree to random selection of attributes at each node of decision tree.

	                                                Average Depth   	Number of nodes
Tree constructed through ID3                         	8.22463768116	        275
Tree constructed through random attribute selection  	10.3910550459     	1841


Random selection algorithm accuracy on test data  
Run   	Accuracy of tree constructed using random attribute selection
1.	47.9%
2.	47.1%
3.	51.6%
4.	48.75%
5.	48.35%

Accuracy of ID3 decision tree on test data=  75.85 %
Assumptions
	The assumptions are same as the ID3 algorithm code submitted earlier which are:
1.	The tree is traversed to left when zero is input and to right when one is input.
2.	The first attribute found is used for classification when more than one attribute has same information gain.
3.	If there is only an instance of data left, the class of instance is value for the node in tree.
