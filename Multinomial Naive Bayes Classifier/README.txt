-----------------------------------------------------------------------------------------------------------------------------------------------------------

MNB.py:

Usage:  MNB.py <training data location> <test data location>

sample-training/test-data-location:http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz
-------------------------------------------------------------------------------------------------------------------------------------------------------------

Aim:-Use multinomial naive bayes classifier to classify news articles into different categories 

Assumptions:
1.	Special characters do not define the class hence a special character or group of special characters is dropped. 
2.	Similarly, the special characters preceding and following a word does not help in defining a class hence are dropped.
3.	The case of the word is ignored in storing and classification.
Implementation Details:
1.	Five folders were randomly chosen from train data folder for training and corresponding five folders were chosen from test.
2.	From every file the list of works is extracted using space or new line as delimiter.
3.	The header potion of the file is ignored (from start to Lines: number).
4.	The words are the cleaned based on the assumptions mentioned above and the stop words are removed.
5.	The words are stored in a dictionary of dictionaries of the following format:
{class1: {word11:count11, word12:count12………………}
class2: {word21:count21, word22:count22………………}}
6.	For each test data the class is predicted, and error is incremented if predicated class is different from the actual class.
7.	An accuracy of over 95% was observed consistently.

