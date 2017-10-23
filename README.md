# Linear Regression, K-means clustering, Gaussian RBF #

This project was implemented as part of an assignment for the course __*CSE574 : Introduction to Machine Learning*__ at _University at Buffalo, The State University of New York_ in Fall 2016. The goal of the project was to develop a Linear Regression system to perform fine grained ranking of webpages.

### Dataset ###

The developed system was trained on [LETOR 4.0 Dataset](https://www.microsoft.com/en-us/research/project/letor-learning-rank-information-retrieval/). The \'Querylevelnorm.txt\’ from MQ2007 was used for this project. The dataset consists of 69623 samples and 46 features with rankings as 0, 1 or 2. This dataset is divided into three sets of 76% of the samples, 12 % of the samples and the remaining for Training, Validation and Testing respectively.

### Implementation ###

#### Data Pre-processing ####

The implementation of the project involved performing dimensionality reduction and then training the system to learn to rank.
Dimensionality reduction is performed by performing clustering in the form of K-means clustering and then applying Gaussian Radial Basis Function over the identified centroids to reduce the dimensions.

The samples are then randomly shuffled to overcome the stagnation of Stochastic Gradient Descent at a local optima and improve convergence.

#### Linear Regression ####

Three implementations of Linear Regression was developed using
* Gradient Descent (GD)
> A Linear Regression system using gradient descent algorithm was developed to learn the rankings of the pages. A system to adapt the learning rate was developed to optimize the learning rate which penalized the learning rate by two-thirds if the at any iteration the cost of the system increased and reward the learning rate marginally if the cost of the system decreased. This optimization was carried out to speed up the running time of the system.
* Stochastic Gradient Descent (SGD)
* Closed Form Solution

Optimizations of the hyperparameters was carried out by iteratively searching in the hyperparameter space for the optimal solution. Hyperparameters tuned include the number of basis functions to reduce the dimensions, the learning rate, L2 regularization parameters.

### Results ###

The results were calculated in the form of error. The E<sub><i>rms</i></sub> observed over the validation set and test set are

   System | Validation Error | Test Error
   -------|---------|---------
   GD | 0.5832 | 0.5681
   SGD | 0.5743 | 0.5566
   Closed Form | 0.5736 | 0.5565
   
   On analysis of the dataset, it is observed that 51632 samples\(roughly 74%) have 0 as the output rank and 14128 samples\(roughly 20%) have 1 as the output rank and the rest have 2 as the rank. This skewness in the ground truths is causing the system to perform poorly.
