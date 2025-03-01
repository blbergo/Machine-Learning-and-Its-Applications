#-------------------------------------------------------------------------
# AUTHOR: Bryan Bergo
# FILENAME: decision_tree_2.py
# SPECIFICATION: Predicts the class of a contact lens using a decision tree algorithm
# FOR: CS 4210- Assignment #2
# TIME SPENT: 20 minutes
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#Importing some Python libraries
from sklearn import tree
import csv

dataSets = ['./assignment-2/data/contact_lens_training_1.csv', './assignment-2/data/contact_lens_training_2.csv', './assignment-2/data/contact_lens_training_3.csv']

for ds in dataSets:

    dbTraining = []
    X = []
    Y = []

    #Reading the training data in a csv file
    with open(ds, 'r') as csvfile:
         reader = csv.reader(csvfile)
         for i, row in enumerate(reader):
             if i > 0: #skipping the header
                dbTraining.append (row)

    #Transform the original categorical training features to numbers and add to the 4D array X.
    #For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3, X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
    feature_map = {'Young': 1, 'Prepresbyopic': 2, 'Presbyopic': 3, 'Myope': 1, 'Hypermetrope': 2, 'No': 1, 'Yes': 2, 'Reduced': 1, 'Normal': 2, 'Soft': 1, 'Hard': 2}
    
    for instance in dbTraining:
        X.append([feature_map[instance[0]], feature_map[instance[1]], feature_map[instance[2]], feature_map[instance[3]]])
    
    #Transform the original categorical training classes to numbers and add to the vector Y.
    #For instance Yes = 1 and No = 2, Y = [1, 1, 2, 2, ...]
    for instance in dbTraining:
        Y.append(feature_map[instance[4]])

    accuracy_scores = 0
    #Loop your training and test tasks 10 times here
    for i in range (10):

       #Fitting the decision tree to the data setting max_depth=3
       clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=5)
       clf = clf.fit(X, Y)

       #Read the test data and add this data to dbTest
       with open('./data/contact_lens_test.csv', 'r') as csvfile:
           reader = csv.reader(csvfile)
           dbTest = []
           for i, row in enumerate(reader):
               if i > 0:
                     dbTest.append(row)
         
       correct = 0
       for data in dbTest:
           #Transform the features of the test instances to numbers following the same strategy done during training,
           #and then use the decision tree to make the class prediction. For instance: class_predicted = clf.predict([[3, 1, 2, 1]])[0]
           #where [0] is used to get an integer as the predicted class label so that you can compare it with the true label
           class_predicted = clf.predict([[feature_map[data[0]], feature_map[data[1]], feature_map[data[2]], feature_map[data[3]]]])[0]

           #Compare the prediction with the true label (located at data[4]) of the test instance to start calculating the accuracy.
           if feature_map[data[4]] == class_predicted:
               correct += 1
               
       accuracy_scores += correct/len(dbTest)
       
    #Find the average of this model during the 10 runs (training and test set)
    final_accuracy = accuracy_scores/10

    #Print the average accuracy of this model during the 10 runs (training and test set).
    #Your output should be something like that: final accuracy when training on contact_lens_training_1.csv: 0.2
    print('final accuracy when training on ' + ds + ': ' + str(final_accuracy))
 




