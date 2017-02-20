import numpy as np
import sys

class GradientDescent(object):
    #Abstract Class for implementation of Gradient Descent
    #Extends Class object
    def __init__(self):
        pass

    def computeCost(self,X,t,w,gamma):
        #Abstract Method
        #Method's purpose is to compute the cost of the system with the given inputs.
        #Inputs:
        #   X       ->  Input Features. Can be a single array of input features or a matrix.
        #   t       ->  Target values
        #   w       ->  Weights of the system.
        #   gamma   ->  Regularization constant
        #Outputs:
        #   J       ->  Cost of the system with the given inputs.

        #Abstract Method defined by trying to get the value of an undeclared variable and then handling the exception.
        try:
            Abstract                                                        #Undeclared Variable. Raises a NameError Exception
        except NameError:                                                   #Handling the NameError Exception
            print("Abstract Method not Implemented: %s"%('computeCost()'))
            sys.exit(0)

    def gradientDescent(self,X,t,w,gamma,beta):
        #Abstract Method
        #Method's purpose is to compute the magnitude and direction with which the weights
        #are to be changed to optimize the system for the least cost.
        #Inputs:
        #   X       ->  Input Features. Can be a single array of input features or a matrix.
        #   t       ->  Target values
        #   w       ->  Weights of the system.
        #   gamma   ->  Regularization constant
        #   beta    ->  The precision value. If two successively computed costs are equivalent upto this precision value, the
        #               gradient descent algorithm stops.
        #Outputs:
        #   [J,w]   ->  A list containing a list of costs and the final weights.

        #Method abstraction achieved by raising a NameError Exception by fetching the value of an undeclared variable
        #and then handling the exception. Exception is overcome when method overridden in the subclasses.
        try:
            Abstract                                                        #Undeclared Variable. Raises a NameError Exception
        except NameError:                                                   #Handling the NameError Exception
            print("Abstract Method not Implemented: %s"%('gradientDescent()'))
            sys.exit(0)

    def error(self,X,t,w,gamma):
        #Abstract Method
        #Method's purpose is to compute the error of the system over the validation and test set.
        #Inputs:
        #   X       ->  Input Features. Can be a single array of input features or a matrix.
        #   t       ->  Target values
        #   w       ->  Weights of the system.
        #   gamma   ->  Regularization constant
        #Outputs:
        #   J       ->  Cost of the system with the given inputs.

        #Abstract Method defined by trying to get the value of an undeclared variable and then handling the exception.
        try:
            Abstract                                                        #Undeclared Variable. Raises a NameError Exception
        except NameError:                                                   #Handling the NameError Exception
            print("Abstract Method not Implemented: %s"%('error()'))
            sys.exit(0)


class BatchGradientDescent(GradientDescent):
    #Class to Implement Batch Gradient Descent.
    #Extends the Abstract Class Gradient Descent.
    #Considers the complete set of inputs as a single batch and performs computation on the batch.
    def __init__(self):
        super(BatchGradientDescent,self).__init__()
        self.alpha = 1                                                      #Initializing the learning rate of the system to 1.

    def computeCost(self,X,t,w,gamma):
        #Computes the cost which is the sum of squared difference between the actual and the predicted value
        #Inputs:
        #   X       ->  Matrix of the Input Records/Feature Records with a 1 appended to the begining of the Matrix to account for the bias
        #   t       ->  The actual target values
        #   w       ->  The weights for the linear regression system
        #   gamma   ->  Regularization constant
        #Outputs:
        #   J       ->  The cost of the system with weights w
        m = len(t)                                                          # 'm' has the number of records in the dataset
        J = sum(np.array(np.dot(X,w) - t).__pow__(2))/(2*m)                 #Unregularized cost
        temp_w = np.copy(w)                                                 #Copying the weights to perform regularization computations
        temp_w[0][0] = 0                                                    #Setting the first weight to 0 since w0 is the bias weight.
                                                                            #Done to regularize only the input features.
        J = J + gamma*np.dot(temp_w.transpose(),temp_w)/(2*m)               #Regularized cost
        return J
        #J(w) or J is the cost of the system when it works with the weights w
        #A larger J indicates a larger error in the predicted values. Our goal is to reduce the cost of the system
        #by optimizing the weights or the parameters of the system.

    def gradientDescent(self,X,t,w,gamma,beta):
        #Computes the magnitude and direction with which the weights are to be changed to optimize the system for the
        #least cost.
        #Inputs:
        #   X       ->  Matrix of the Input Records/Feature Records with a 1 appended to the begining of the Matrix to account for the bias
        #   t       ->  The actual target values
        #   w       ->  The weights for the linear regression system
        #   gamma   ->  Regularization constant
        #   beta    ->  The precision value. If two successively computed costs are equivalent upto this precision value, the
        #               gradient descent algorithm stops.
        #Outputs:
        #   [J,w]   ->  A list containing a list of costs and the final weights.
        m = len(t)                                                          #'m' indicates the number of records in the dataset
        J = [self.computeCost(X,t,w,gamma)]                                 #list of costs initialized
        print('Initial Cost: %s'%J[-1])
        flag = False                    #Flag value initialized to False. Indicates that two successively computed costs aren't equivalent
                                        #upto the indicated precision value. True here, indicates that the computation loop must end since
                                        #two successive computations have the same value upto the indicated precision.
        inc_rate = 1.00005              #Increment Rate of 0.005%. Used to reward the system in case of successful computation
        prev_alpha = None               #prev_alpha initialized to None indicating no previous alpha value exists
        while not flag:
            gradient = self.alpha*np.dot(X.transpose(),(np.dot(X,w) - t))/m     #Unregularized gradient
            temp_w = np.copy(w)
            temp_w[0][0] = 0
            gradient = gradient + (gamma*temp_w)/m                          #Regularized gradient
            w = w - gradient                                                #Weight Optimization through gradient descent
            interim_J = np.asscalar(self.computeCost(X,t,w,gamma))          #Computing the cost with new weights
            if interim_J > J[-1]:                                           #If the computed cost is higher than the previous, then the
                                                                            # gradient descent algorithm is diverging
                w = w + gradient                                            #Resetting the weights
                if self.alpha == 0.0:                                       #If learning rate is 0, finish computation
                    break
                if prev_alpha is None:                                      #If the previous learning rate was None, penalize heavily
                    self.alpha = self.alpha/3.0                             #Penalizing the learning rate. Decreasing it by a factor of 3
                    inc_rate = 1.00005                                      #Increment Rate set to 0.005% in hopes of improvement after penalization
                else:                                                       #If previous learning rate was stored, penalize the system
                    self.alpha = prev_alpha                                 #by returning the learning rate to this stored value
                    prev_alpha = None                                       #Stored Learning Rate Reset to None to heavily penalize the system
                                                                            #a second time if necessary
                    inc_rate = 1                                            #Increment Rate is set to 0. The System cannot handle increments
                continue                                                    #Continuing to the next iteration without updating the cost
            elif interim_J < J[-1]:                                         #If the descent was successful
                prev_alpha = self.alpha                                     #Store the learning rate for later marginal penalization
                self.alpha = inc_rate*self.alpha                            #Rewarding the system for the successful computation by
                                                                            #increasing learning rate
            J.append(interim_J)                                             #appending the cost computed for later visualization.
            if int(J[-1]/beta) == int(J[-2]/beta):                          #comparing the last two values to check for the end of loop
                flag = True

        return np.array([np.array(J),w])
        #Returned weights are the optimal for the system.

    def error(self,X,t,w,gamma):
        #Computes the error of the validation and test set.
        #Inputs:
        #   X       ->  Matrix of the Input Records/Feature Records with a 1 appended to the begining of the Matrix to account for the bias
        #   t       ->  The actual target values
        #   w       ->  The weights for the linear regression system
        #   gamma   ->  Regularization constant
        #Outputs:
        #   E       ->  The cost of the system with weights w
        E = self.computeCost(X,t,w,0)

        return E


class StochasticGradientDescent(GradientDescent):
    def __init__(self):
        super(StochasticGradientDescent,self).__init__()
        self.alpha = None                                                   #Initializing the learning rate of the system to None.

    def learningRate(self,alpha):
        self.alpha = alpha

    def computeCost(self,X,t,w,gamma):
        #Computes the cost which is the sum of squared difference between the actual and the predicted value
        #Inputs:
        #   X       ->  A single record of input features.
        #   t       ->  The actual target value
        #   w       ->  The weights for the linear regression system
        #   gamma   ->  Regularization constant
        #Outputs:
        #   J       ->  The cost of the system with weights w and input features X
        J = pow(np.asscalar(np.dot(X,w) - t),2)/2                           #Unregularized cost
        temp_w = np.copy(w)                                                 #Copying the weights to perform regularization computations
        temp_w[0][0] = 0                                                    #Setting the first weight to 0 since w0 is the bias weight.
                                                                            #Done to regularize only the input features.
        J = J + gamma*np.dot(temp_w.transpose(),temp_w)/2                   #Regularized cost
        return J
        #J(w) or J is the cost of the system when it works with the weights w
        #A larger J indicates a larger error in the predicted values. Our goal is to reduce the cost of the system
        #by optimizing the weights or the parameters of the system.

    def gradientDescent(self,X,t,w,gamma,beta):
        #Computes the magnitude and direction with which the weights are to be changed to optimize the system for the
        #least cost.
        #Inputs:
        #   X       ->  Matrix of the Input Records/Feature Records with a 1 appended to the begining of the Matrix to account for the bias
        #   t       ->  The actual target values
        #   w       ->  The weights for the linear regression system
        #   gamma   ->  Regularization constant
        #   beta    ->  The precision value. If two successively computed costs are equivalent upto this precision value, the
        #               gradient descent algorithm stops.
        #Outputs:
        #   [J,w]   ->  A list containing a list of costs and the final weights.
        if self.alpha is None:                                              #Check whether learning rate has been set
            print "Error: Learning Rate not set.Ending the training session"    #Display error message
            sys.exit(0)                                                     #Exit system
        m = len(t)                                                          #'m' indicates the number of records in the dataset
        J = 0                                                               #Initializing cost/error to 0
        for i in range(m):                                                  #Iterating over the entire dataset.
            gradient = np.asscalar(np.dot(X[i],w)
                        - t[i])*X[i].reshape((len(X[i]),1))                 #Computing unregularized gradient for each input record
            gradient = gradient + gamma*w                                   #Updating gradient to regularize it
            w = w - self.alpha*gradient                                     #Updating weights
            J += self.computeCost(X[i],t[i],w,gamma)                        #Computing cost of the system
        J = J/m                                                             #Averaging cost over all input records
        return np.array([np.array(J),w])
        #Returned weights are the optimal for the system.

    def error(self,X,t,w,gamma):
        #Computes the error of the validation and test set.
        #Inputs:
        #   X       ->  Matrix of the Input Records/Feature Records with a 1 appended to the begining of the Matrix to account for the bias
        #   t       ->  The actual target values
        #   w       ->  The weights for the linear regression system
        #   gamma   ->  Regularization constant
        #Outputs:
        #   E       ->  The cost of the system with weights w
        m = len(t)                                                          #Length of dataset over which error is to be computed
        J = 0                                                               #Initializing cost/error to 0
        for i in range(m):                                                  #Iterating over all m input sets
            J += self.computeCost(X[i],t[i],w,gamma)                        #Compute Cost for each input record
        return J/m                                                          #Return the averaged cost.


class LinearRegression():
    #Class to implement LinearRegression.
    #Performs LinearRegression using BatchGradientDescent or StochasticGradientDescent based on users choice.
    def __init__(self):
        self.weights = None
        self.gamma = 0                                                      #Initializing the regularization constant to 0.
        self.gradient_descent = GradientDescent()                           #Initializing Gradient Descent Algorithm type to abstract
        self.precision = 0.0001                                             #Required Precision Set.

#Perform SGD within the function.
    def trainSGD(self,learningRate):
        self.gradient_descent = StochasticGradientDescent()
        self.gradient_descent.learningRate(learningRate)                    #Set the learning rate of the regression

    def trainBGD(self):
        self.gradient_descent = BatchGradientDescent()

    def setPrecision(self,precision):
        self.precision = precision

    def trainSystem(self,X,t,gamma=0):
        #Trains the system for the given input features and the target values.
        #Inputs:
        #   X       ->  Input Feature Matrix without bias
        #   t       ->  The target values
        #   gamma   ->  Regularization constant
        #Outputs:
        #   No outputs.
        m = len(t)                                                          #'m' indicates the number of records in the dataset
        self.gamma = gamma                                                  # Value of gamma initialized to indicate Regularization
        X = np.concatenate((np.ones((m,1)),np.array(X)),axis=1)             #Appending 1 to each input feature record as a bias value
        self.weights = np.zeros((X.shape[1],1))                             #Initializing the weights to 0

        #Performing gradient descent to find the optimal weights
        [J,self.weights] = self.gradient_descent.gradientDescent(X,t,self.weights,
                                self.gamma,self.precision)                  #Return Value is a list of costs and weights
        print "Weights: ",self.weights
        print "Final Cost: ",J[-1]

    def validateSystem(self,X,t):
        #Validates the System by computing the RMS Error value.
        #Inputs:
        #   X       ->  Input Feature Matrix of validation set without bias
        #   t       ->  Target Values

        m = len(t)                                                              #'m' indicates the number of records in the dataset
        X = np.concatenate((np.ones((m,1)),np.array(X)),axis=1)                 #Appending 1 to each input feature record as a bias value
        Error = self.gradient_descent.error(X,t,self.weights,self.gamma)        #Computing the Error/Cost for the validation set.
                                                                                #Returned Error is the squared mean value
        E_rms = pow(2*Error,0.5)                                                #Computing the Root Mean Squared Error Value.
        print("Validation,Root Mean Squared Error: %f"%(E_rms))

        return E_rms


    def testSystem(self,X,t):
        #Tests the System by computing the RMS Error value.
        #Inputs:
        #   X       ->  Input Feature Matrix of validation set without bias
        #   t       ->  Target Values

        m = len(t)                                                              #'m' indicates the number of records in the dataset
        X = np.concatenate((np.ones((m,1)),np.array(X)),axis=1)                 #Appending 1 to each input feature record as a bias value
        Error = self.gradient_descent.error(X,t,self.weights,self.gamma)        #Computing the Error/Cost for the validation set.
                                                                                #Returned Error is the squared mean value
        E_rms = pow(2*Error,0.5)                                                #Computing the Root Mean Squared Error Value.
        print("Root Mean Squared Error: %f"%(E_rms))
