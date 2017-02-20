import numpy as np
from numpy.linalg import pinv

class ClosedFormSolution():
    #Class to implement closed form solution for Linear Regression.
    #The entire set of input features is considered as a single batch.

    def computeWeights(self,X,t,gamma):
        #Method to compute the weights of the system by performing closed form solution
        #Inputs:
        #   X       ->  Input feature matrix
        #   t       ->  Target Values (Column Matrix)
        #   gamma   ->  Regularization constant
        #Output:
        #   w       ->  Weights of the system
        (m,n) = X.shape                                                     #Storing the shape of input feature matrix
        w = np.dot(np.dot(pinv(gamma*np.identity(n)+np.dot(X.transpose(),X)),X.transpose()),t)

        return w
        #Returns the weights of the system as calculated
    def error(self,X,t,w,gamma):
        #Method to compute the error of the system by performing closed form solution
        #Inputs:
        #   X       ->  Input feature matrix
        #   t       ->  Target Values (Column Matrix)
        #   w       ->  Weights of the system
        #   gamma   ->  Regularization constant
        #Output:
        #   e       ->  Error of the system
        m = len(t)                                                          #Length of dataset overwhich error is to be calculated
        e = sum(np.array(np.dot(X,w)-t).__pow__(2))/(2*m)

        return e
        #Returns the mean squared error of the system.

class LinearRegression():
    #Class to implement LinearRegression.
    #Performs LinearRegression using BatchGradientDescent or StochasticGradientDescent based on users choice.
    def __init__(self):
        self.weights = None
        self.gamma = 0                                                      #Initializing the regularization constant to 0.

#Perform SGD within the function.
    def trainCFS(self):
        self.closed_form_solution = ClosedFormSolution()

    def trainSystem(self,X,t,gamma=0):
        #Trains the system for the given input features and the target values using the closed form solution
        #Inputs:
        #   X       ->  Input Feature Matrix without bias
        #   t       ->  The target values
        #   gamma   ->  Regularization constant
        #Outputs:
        #   No outputs.
        m = len(t)                                                          #'m' indicates the number of records in the dataset
        self.gamma = gamma                                                  #Value of gamma initialized to indicate Regularization
        X = np.concatenate((np.ones((m,1)),np.array(X)),axis=1)             #Appending 1 to each input feature record as a bias value

        self.weights = self.closed_form_solution.computeWeights(X,
                                t,gamma)                                    #Computing weights of the system using the closed form solution
                                                                            #method.

        print "Weights: ",self.weights

    def validateSystem(self,X,t):
        #Validates the System by computing the RMS Error value.
        #Inputs:
        #   X       ->  Input Feature Matrix of validation set without bias
        #   t       ->  Target Values

        m = len(t)                                                              #'m' indicates the number of records in the dataset
        X = np.concatenate((np.ones((m,1)),np.array(X)),axis=1)                 #Appending 1 to each input feature record as a bias value
        Error = self.closed_form_solution.error(X,t,self.weights,self.gamma)    #Computing the Error/Cost for the validation set.
                                                                                #Returned Error is the squared mean value
        E_rms = pow(2*Error,0.5)                                                #Computing the Root Mean Squared Error Value.
        print("Root Mean Squared Error: %f"%(E_rms))

        return Error


    def testSystem(self,X,t):
        #Tests the System by computing the RMS Error value.
        #Inputs:
        #   X       ->  Input Feature Matrix of validation set without bias
        #   t       ->  Target Values

        m = len(t)                                                              #'m' indicates the number of records in the dataset
        X = np.concatenate((np.ones((m,1)),np.array(X)),axis=1)                 #Appending 1 to each input feature record as a bias value
        Error = self.closed_form_solution.error(X,t,self.weights,self.gamma)    #Computing the Error/Cost for the validation set.
                                                                                #Returned Error is the squared mean value
        E_rms = pow(2*Error,0.5)                                                #Computing the Root Mean Squared Error Value.
        print("Root Mean Squared Error: %f"%(E_rms))
