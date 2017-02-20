import numpy as np

class KMeans():
    #Class to compute the kmeans clustering of given dataset.

    def __init__(self):
        #Class initialization method
        pass
    def __distance(self,v,w):
        #Method to calculate the distance between two vectors
        #Inputs:
        #   v,w     ->          2 1D vectors
        #Outputs:
        #   d       ->          euclidean distance between the two vectors v and w
        mag = v - w                                                         #Magnitude of distance between data points in each dimension
        dist = np.sum(mag*mag,axis=1)**0.5                                  #Vectorized implementation of distance calculation
        dist = dist.reshape((len(dist),1))
        return dist
        #Return value is a scalar value that signifies the distance between the two vectors.

    def __partitionData(self,data,centroids):
        #Method to partition data into the respective clusters signified by the centroids by computing the distance between
        #the data points and the cluster centroids. A data point belongs to the cluster when the distance between it and the
        #centroid point for that cluster is the least among the distance with all other clusters.
        #Inputs:
        #   data        ->      Input data
        #   centroids   ->      Cluster centroids
        #Outputs:
        #   clusters   ->      computed centroids of the new clusters
        k = len(centroids)                                                  #Storing the number of centroids
        (m,n) = data.shape
        clusters = np.empty((m,1))
        c = []
        for i in range(k):
            c.append([])                                                    #Initializing a array datastructure to store cluster data points
            clusters = np.concatenate((clusters,
                        self.__distance(data,centroids[i])),axis=1)         #Computing the distance wrt each cluster and creating a matrix
        index = clusters[:,1:].argmin(axis=1).transpose()                   #Determining the cluster with least distance for each data point
        #Rearranging the data points according to the determined clusters
        for i in range(m):
            c[index[i]].append(data[i])
        clusters = np.array([np.array(cluster) for cluster in c])           #Converting the clusters into a numpy array
        return clusters

    def __centroids(self,clusters):
        #Methods to compute the centroids from the clusters.
        #Inputs:
        #   clusters        ->      Clustered Data
        #Outputs:
        #   centroids       ->      Cluster centroids
        centroids = []
        for i in range(len(clusters)):                                      #Iterating over each cluster
            centroids.append(np.mean(clusters[i],axis=0))                   #Computing the centroids of the cluster
        centroids = np.array(centroids)                                     #Converting the centroids list to a numpy array
        return centroids

    def __identical(self,c1,c2):
        #Method to check if two centroids are identical
        #Inputs:
        #   c1,c2       ->  Centroids which are to be checked for identicalness
        #Outputs:
        #   Boolean     ->  A boolean value indicating whether the centroids are identical or not.
        if sum(c1.flatten() == c2.flatten()) == c1.size:                    #sum of true values(1) compared with size of centroids
            return True                                                     #Centroids are identical if sum of true values is equal to size
                                                                            #Implies that all elements are the same in both the centroids
        return False                                                        #Return false if the sum is not equivalent to the size.
                                                                            #Implies that one or more of the elements are different

    def cluster(self,data,number):
        #Method to perform clustering of the given dataset into the mentioned number of clusters
        #Inputs:
        #   data        ->  Input data
        #   number      ->  Number of clusters the data has to be partitioned into
        #Outputs:
        #   clusters    ->  Clustered Data.
        np.random.shuffle(data)                                             #Shuffling data randomly before clustering
        centroids = []                                                      #List to store the centroids
        for i in range(number):
            centroids.append(data[i])                                       #Initializing the initial cluster centroids
        centroids = np.array(centroids)                                     #Converting the list to a numpy array list
        flag = False                                                        #Initializing loop flag to False. flag signifies if two computed
                                                                            #centroids are identical
        while not flag:                                                     #Iterate till the centroids are not identical
            c = centroids                                                   #Making a copy of centroids for later comparision
            clusters = self.__partitionData(data,centroids)                 #Parititioning data into clusters
            centroids = self.__centroids(clusters)                          #Computing the centroids of the clusters
            flag = self.__identical(c,centroids)                            #Updating the flag about the identicalness of the two centroids
        return [clusters,centroids]                                         #Returning the clusters and centroids obtained by K-Means Clustering
