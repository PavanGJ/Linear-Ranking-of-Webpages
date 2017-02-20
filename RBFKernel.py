import numpy as np

class GaussianRBF():
    #Class to implement Gaussian Radial Basis Function.
    #It's primary purpose is to convert an input vector to a corresponding scaled scalar value.
    def __init__(self):
        #Class initialization.
        pass

    def __basisFunction(self,X,center,epsilon):
        #Method to compute the Gaussian Radial Basis Function given an input vector and corresponding basis centers.
        #Basis Function is being applied to reduce the dimensionality of the vector
        #Inputs:
        #   X       ->  Input Feature vector
        #   center  ->  Basis Function Centers.
        #Outputs:
        #   S       ->  Scalar value corresponding to the input vector and center
        epsilon = epsilon.reshape((epsilon.shape[0],1))
        M = np.sum(np.dot(((X-center)*(X-center)),epsilon),axis=1)          #Computing the squared distance of the two vectors X and center
        S = np.exp(-1*M/2)                                                  #Computing the basis function

        return S.reshape((len(S),1))
        #S is a column matrix which represents the distance of input features from the center given.

    def computeRBF(self,X,centers,epsilon):
        #Method to compute the Gaussian Radial Basis Function given an input matrix containing clustered vectors.
        #Radial Basis Function reduces the dimensionality of the Input Feature Matrix from D input Features to M Basis Functions
        #where D is the number of features and M is the number of clusters.
        #Inputs:
        #   X           ->  Clustered Input Feature Matrix
        #   centers     ->  cluster centers (centroids)
        #                       centers     =   [center 1, centers 2, center 3, ...., center D]
        #                       each center corresponds to a cluster.
        #Output:
        #   P       ->  DxN dimensional Basis Function Matrix
        #               D   ->  Number of Basis Function
        #               N   ->  Number of data in dataset
        (m,n) = X.shape                                                     #Storing the shape of input feature matrix
        P = np.empty((m,1))                                                 #Initializing an empty list to store the basis functions
        for i in range(len(centers)):                                       #Iterating through the centers
            p = self.__basisFunction(X,centers[i],epsilon[i])               #calculating the basis function for each center
            P = np.concatenate((P,p),axis=1)                                #Appending the basis function of all data points to the final matrix
        return np.array(P[:,1:])
