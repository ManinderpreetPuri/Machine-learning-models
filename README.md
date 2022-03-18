# Machine-learning-models

Neural Network Models for MNIST data set:

The MNIST database of handwritten digits (from 0 to 9) has a training set of 55,000 
examples, and a test set of 10,000 examples. The digits have been size-normalized and 
centered in a fixed-size image (28x28 pixels) with values from 0 to 1. I used the 
following code with TensorFlow in Python to download the data.

from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

Every MNIST data point has two parts: an image of a handwritten digit and a  corresponding label. We will call the images 'x' and the labels 'y'. Both the training set and test set contain 'x' and 'y'. Each image is 28 pixels by 28 pixels and can be flattened into a vector of 28x28 = 784 numbers.

As mentioned, the corresponding labels in the MNIST are numbers between 0 and 9, describing which digit a given image is of. In this project, we regard the labels as 
one-hot vectors, i.e. 0 in most dimensions, and 1 in a single dimension. In this case, the n-th digit will be represented as a vector which is 1 in the n dimensions. For example, 3 would be [0,0,0,1,0,0,0,0,0,0]. The project aims to build NNs for classifying handwritten digits in the MNIST database, train it on the training set and test it on the test set. 

1. In the project, the performance of a NN is measured by the its prediction accuracy in classifying images from the test set, i.e. number of the correctly predicted images / number of the images in the test set.
2. I have created THREE NNs by changing the architecture. I changed the number of layers, use different type of layers, and tried various activation layers.
3. I trained and tested NNs with different parameter setting, e.g. learning rate.
4. Report contain the following content
a. Architectures of the NNs, with figures;
b. Experiments and performances, with parameter setting;
c. Discussion on the improvement / deterioration of the NN’s performance after changing the architecture and parameter setting;
d. The best performance of the NNs;

Prediction of median value of House in Boston using Regression modles:

In this project I developed 3 end to end regression models: SVM, Linear Regression and K-Nearest Neighbors. The models are developed to accurately predict median value 
of Boston houses in any given suburb, where the suburbs are described based on their provided characteristics:
o CRIM per capita crime rate by town
o ZN proportion of residential land zoned for lots over 25,000 sq.ft.
o INDUS proportion of non-retail business acres per town
o CHAS Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
o NOX nitric oxides concentration (parts per 10 million)
o RM average number of rooms per dwelling
o AGE proportion of owner-occupied units built prior to 1940
o DIS weighted distances to five Boston employment centres
o RAD index of accessibility to radial highways
o TAX full-value property-tax rate per $10,000
o PTRATIO pupil-teacher ratio by town
o B 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
o LSTAT % lower status of the population
o MEDV Median value of owner-occupied homes in $1000's
Therefore, the 13 first columns would generate X, and the last column (MEDV ) would be the label. The dataset is from UCI Machine Learning repository, at: 
http://lib.stat.cmu.edu/datasets/boston).

The code:
ü Loads the dataset (used this library : from pandas import read_csv)
ü Describes the data: printing the shape of the dataset, the type of data, and related correlation correlation 
ü Visualizes the data (with pyplot)
ü Splits the data into train and test (20% of the entire data for test). 
ü Trains, tests and compares the performance of the three developed models

Report:
ü Describes each model. The basic of SVM, Linear Regression and K-Nearest Neighbors. Explains what steps I have taken for fine tuning the model (changing the parameters). 
ü Compares the performance of each model individually using different parameters;  then compare the performance of the best of each model with other models.
