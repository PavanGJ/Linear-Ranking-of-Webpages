import numpy as np

class LeToR():
    #Class to read LeToR data and perform initial data processing.
    def __init__(self):
        #Class initialization method
        pass

    def parse(self,file_path):
        #Method to read and parse the data from the input file
        #Inputs:
        #   file_path       ->      Input File Path
        #Outputs:
        #   None
        raw_data = open(file_path,'r')                                      #Reading the input file
        self.data = []                                                      #Initializing empty list to store parsed data
        #Parsing data to extract features and targets.
        #Format of row in data:        [t qid:n 1:x1 2:x2 .... i:xi #docid = zzzzzz inc = zzzzz prob = zzzzz]
        #Data to be extracted:      [t x1 x2 ... xi]
        for row in raw_data:
            temp = row.split('#')[0].split(':')
            t = np.array([float(temp[0].split()[0])])
            x = np.array([float(temp[i].split()[0]) for i in range(2,len(temp))])

            self.data.append(np.concatenate((t,x)))
        self.data = np.array(self.data)
        np.random.shuffle(self.data)                                        #Shuffling the data for randomness
        #self.data contains the entire data set with the first element in each row as the target and the rest being features

    def getTrainingSet(self):
        #Method to return the training set records
        end = round(0.76*self.data.shape[0])
        m = end
        return [self.data[:end,0].reshape((m,1)),self.data[:end,1:]]

    def getValidationSet(self):
        #Method to return the validation set records
        begin = round(0.76*self.data.shape[0])
        end = begin + round(0.12*self.data.shape[0])
        m = end - begin
        return [self.data[begin:end,0].reshape((m,1)),self.data[begin:end,1:]]
    def getTestSet(self):
        #Method to return the test set records
        begin = round(0.76*self.data.shape[0]) + round(0.12*self.data.shape[0])
        m = self.data.shape[0] - begin
        return [self.data[begin:,0].reshape((m,1)),self.data[begin:,1:]]
