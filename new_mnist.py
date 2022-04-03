import pandas as pd
import numpy as np
import math
from matplotlib import pyplot as plt
from imp import reload
from sklearn.metrics import zero_one_loss

#To calculate the Polynomial Kernel
def poly_ker(x1, x2, power):
  
  return (1+np.dot(x1,x2))**power

#To calculate the Kernel/gram Matrix
def kernelMatrix(X1, X2, power):
    
  x1_samples = X1.shape[0]
  x2_samples = X2.shape[0]
  xA1 = X1.to_numpy()
  xA2 = X2.to_numpy()
  K = np.zeros((x1_samples, x2_samples))
  for i in range(x1_samples):
    for j in range(x2_samples):
      K[i,j] = poly_ker(xA1[i,:], xA2[j,:], power)
  
  if xA1.shape == xA2.shape : 
      K = K + K.T - np.diag(K.diagonal())    
      return K
  else:
      return K

#To calculate the zero_one_loss  
def Z_O_L(y_test,y_pred):
      errors = []
      accuracy = []
      zero_one_error = []
      #test for accuracy:
      error = 0
      #y_testNP = y_test.to_numpy()
      #y_predNP = np.array(y_pred)
      for i in range(len(y_pred)):
          if y_pred[i] != y_test[i]:
              error += 1
              
      errors.append(error)
      accuracy = 1 - error/len(y_pred)
      zero_one_error = error/len(y_pred)
      return zero_one_error
      
class KernelPerceptron(object):

  def __init__(self, label, epoch, Kernel_Matrix, power):
    self.epoch = epoch
    self.label = label
    self.KM = Kernel_Matrix
    self.power = power
    
    
  def set_label(self, y):
    
    return np.where(y == self.label, 1, -1)
  
  def train_smallest_error(self, X_train, y_train):
    
    n_samples, n_features = X_train.shape
    alpha = np.zeros(n_samples)
    smallest_error = 1
    y_train = self.set_label(y_train)
    self.errors = []
    self.list_epoch = []
    self.list_alphas = []
    km = self.KM(X_train,X_train,self.power)
    
      
    for e in range(self.epoch):
      error = 0
      y_pred = []
      for i in range(n_samples):
        y_hat = np.sign(np.sum(alpha*y_train*km[:, i]))
        y_pred.append(y_hat)
        if y_hat != y_train[i]:
          alpha[i] += 1
          error += 1
      print("::::::In Training_Smallest_Error::::::")    
      print("In Epoch: ",e+1)    
      print("Zero One loss :",Z_O_L(y_train,y_pred))
      
      self.list_epoch.append(e)    
      self.errors.append(error) 
      self.list_alphas.append(alpha)
     
    smallest_error = self.errors[np.argmin(self.errors)]
    smallest_epoch = self.list_epoch[np.argmin(self.errors)]
    smallest_alpha = self.list_alphas[np.argmin(self.errors)]
    print("Smallest Error: ",smallest_error)
    print("Epoch with smallest error:",smallest_epoch)
    print("Smallest alpha: ",smallest_alpha)
    
    si = np.nonzero(smallest_alpha)
    self.sva = smallest_alpha[si]
    self.svx = X_train.iloc[si]
    self.svy = y_train[si]

  def train_average_predictor(self, X_train, y_train):
    
    n_samples = X_train.shape[0]
    n_features = X_train.shape[1]
    alpha = np.zeros(n_samples)
    y_train = self.set_label(y_train)
    error = 0
    ensemble = []
    km = self.KM(X_train,X_train,self.power)
    
    for e in range(self.epoch):
      y_pred = []
      for i in range(n_samples):
        y_hat = np.sign(np.sum(alpha*y_train*km[:, i]))
        y_pred.append(y_hat)
        if y_hat != y_train[i]:
          alpha[i] += 1
      print("::::::In Training_Average_Predictor::::::")    
      print("Epoch: ",e+1)     
      print("zero One loss =",Z_O_L(y_train,y_pred))   
      
      ensemble.append(alpha)
    
    avg_ensemble = np.sum(ensemble,axis=0)/(self.epoch + 1)
    ensemble = np.array(avg_ensemble)
    print(avg_ensemble)
    print(ensemble)    
    si = np.nonzero(ensemble)
    for k in range(n_samples):
      y_hat = np.sign(np.sum(ensemble*y_train*km[:, k]))
      if y_hat != y_train[k]:
        error +=1
    accuracy = 1 - (float(error)/n_samples)
    print("Training Accuracy",accuracy)
    

    self.sva = ensemble[si]
    self.svx = X_train.iloc[si]
    self.svy = y_train[si]
      
  def predict(self, X_test, power, Matrix_predict = None, toLabel = True):
    
    n_samples = X_test.shape[0]
    y_hat = np.zeros(n_samples)
    if Matrix_predict is None:
      Matrix_predict = kernelMatrix(X_test, self.svx, self.power)
    for i in range (n_samples):
        y_hat[i] = np.sum(self.sva*self.svy*Matrix_predict[i, :])
    
    return np.where(y_hat >= 0, self.label, -1) if toLabel else y_hat
   

       
       
       
if __name__ == "__main__":
    
    def gen_non_lin_separable_data():
        #importing the datasets
        training_images=pd.read_csv('./mnist_train.csv',sep=',')
        testing_images=pd.read_csv('./mnist_test.csv',sep=',')
        
        #separating the training label and data of the dataset and label encoding
        Y_train = training_images['label']
        X_train = training_images.drop(['label'],1)
        #Y_train = to_categorical(Y_train)

        #separating the test label and data of the dataset and label encoding
        Y_test = testing_images['label']
        X_test = testing_images.drop(['label'],1)

        # shuffle dataset
        index = np.random.RandomState(seed = 256).permutation(X_train.index)
        X_train = X_train.reindex(index)
        Y_train = Y_train[index]
 
        
        return X_train, Y_train, X_test, Y_test

    def split_train(X_train, Y_train):
        x_train = X_train[0:1500]
        y_train = Y_train[0:1500]
        return x_train, y_train

    def split_test(X_test, Y_test):
        x_test = X_test[0:500]
        y_test = Y_test[0:500]
        return x_test, y_test


    def test_kernel_degree_avg():
        X_train, Y_train, X_test, Y_test = gen_non_lin_separable_data()
        x_train, y_train = split_train(X_train, Y_train)
        x_test, y_test = split_test(X_test, Y_test)
        ZOL = []
        YT = y_test.to_numpy()
        for degree in range(2,7):
            perceptrons = []
            for i in range (10):
                kp = KernelPerceptron(i,epoch = 10, Kernel_Matrix=kernelMatrix, power = degree)
                kp.train_average_predictor(x_train, y_train)
                perceptrons.append(kp)           

            y_hat_perceptrons = np.zeros((x_test.shape[0], len(perceptrons)))
            
            for i, kernel in enumerate(perceptrons):
                print(i) #just for tests
                y_hat_perceptrons[:, i] = kernel.predict(x_test, degree, toLabel = False) #calls method predict for all perceprtons
                #print(y_hat_perceptrons)
            one_vs_all = np.argmax(y_hat_perceptrons, axis = 1) #y_hat is a matrix with the y of each perceptron per column. axis = 1 takes the maximum on the column for each prediction in each row
            print("one vs all:",one_vs_all)
            print(one_vs_all.shape)
            y_predict = np.array([perceptrons[i].label for i in one_vs_all])
            print(y_predict.shape)
            print("Zero_One_Loss for prediction : ",Z_O_L(YT,y_predict) )
            ZOL.append(Z_O_L(YT,y_predict))
            
        print(ZOL) 
        plt.plot([2,3,4,5,6], ZOL)
        plt.xticks([2,3,4,5,6])
        plt.ylabel("Zero-One Loss")
        plt.xlabel("Degree") 
        plt.title('Degree Average Predictor ')

            
    def test_kernel_degree_smallest():
        
        X_train, Y_train, X_test, Y_test = gen_non_lin_separable_data()
        x_train, y_train = split_train(X_train, Y_train)
        x_test, y_test = split_test(X_test, Y_test)
        ZOL = []
        
        YT = y_test.to_numpy()            
        for degree in range(2,7):
            perceptrons = []
            for i in range (10):
                 kp = KernelPerceptron(i,epoch = 10, Kernel_Matrix=kernelMatrix, power = degree)
                 kp.train_smallest_error(x_train, y_train)
                 perceptrons.append(kp)         
           
            y_hat_perceptrons = np.zeros((x_test.shape[0], len(perceptrons)))
            
            for i, kernel in enumerate(perceptrons):
                print(i) #just for tests
                y_hat_perceptrons[:, i] = kernel.predict(x_test, degree, toLabel = False) #calls method predict for all perceprtons
                #print(y_hat_perceptrons)
            one_vs_all = np.argmax(y_hat_perceptrons, axis = 1) #y_hat is a matrix with the y of each perceptron per column. axis = 1 takes the maximum on the column for each prediction in each row
            print("one vs all:",one_vs_all)
            print(one_vs_all.shape)
            y_predict = np.array([perceptrons[i].label for i in one_vs_all])
            print(y_predict.shape)
            print("Zero_One_Loss for prediction : ",Z_O_L(YT,y_predict) )
            ZOL.append(Z_O_L(YT,y_predict))
            
        print(ZOL)             
        plt.plot([2,3,4,5,6], ZOL)
        plt.xticks([2,3,4,5,6])
        plt.ylabel("Zero-One Loss")
        plt.xlabel("Degree")
        plt.title('Degree Smallest Error')

            

            
    def test_kernel_epoch_AVG():
        X_train, Y_train, X_test, Y_test = gen_non_lin_separable_data()
        x_train, y_train = split_train(X_train, Y_train)
        x_test, y_test = split_test(X_test, Y_test)
        YT = y_test.to_numpy()
        ZOL = [] 

        degree = 2     
        
        # train and predict using average predictor
        for epoch in range(1,11):
            print("epoch:", epoch)
            perceptrons = []

            for i in range (10):
                 kp = KernelPerceptron(i,epoch, Kernel_Matrix=kernelMatrix, power = degree)
                 kp.train_average_predictor(x_train, y_train)
                 perceptrons.append(kp)         
           
            y_hat_perceptrons = np.zeros((x_test.shape[0], len(perceptrons)))
            
            for i, kernel in enumerate(perceptrons):
                print(i) #just for tests
                y_hat_perceptrons[:, i] = kernel.predict(x_test, degree, toLabel = False) #calls method predict for all perceprtons
                #print(y_hat_perceptrons)
            one_vs_all = np.argmax(y_hat_perceptrons, axis = 1) #y_hat is a matrix with the y of each perceptron per column. axis = 1 takes the maximum on the column for each prediction in each row
            print("one vs all:",one_vs_all)
            print(one_vs_all.shape)
            y_predict = np.array([perceptrons[i].label for i in one_vs_all])
            print(y_predict.shape)
            print("Zero_One_Loss for prediction : ",Z_O_L(YT,y_predict) )
            ZOL.append(Z_O_L(YT,y_predict))
            
        print(ZOL)             
        plt.plot([1,2,3,4,5,6,7,8,9,10], ZOL)
        plt.xticks([1,2,3,4,5,6,7,8,9,10])
        plt.ylabel("Zero-One Loss")
        plt.xlabel("Epochs")
        plt.title('Epoch Average Predictor')   
        
        #tain and predict using smallest error predictor   
        
    def test_kernel_epoch_SMERR():        
        X_train, Y_train, X_test, Y_test = gen_non_lin_separable_data()
        x_train, y_train = split_train(X_train, Y_train)
        x_test, y_test = split_test(X_test, Y_test)
        YT = y_test.to_numpy()
        ZOL = [] 

        degree = 2 
          
        for epoch in range(1,11):
            perceptrons = []
            print("epoch:", epoch)
            for i in range (10):
                 kp = KernelPerceptron(i,epoch, Kernel_Matrix=kernelMatrix, power = degree)
                 kp.train_smallest_error(x_train, y_train)
                 perceptrons.append(kp)         
           
            y_hat_perceptrons = np.zeros((x_test.shape[0], len(perceptrons)))
            
            for i, kernel in enumerate(perceptrons):
                print(i) #just for tests
                y_hat_perceptrons[:, i] = kernel.predict(x_test, degree, toLabel = False) #calls method predict for all perceprtons
                #print(y_hat_perceptrons)
            one_vs_all = np.argmax(y_hat_perceptrons, axis = 1) #y_hat is a matrix with the y of each perceptron per column. axis = 1 takes the maximum on the column for each prediction in each row
            print("one vs all:",one_vs_all)
            print(one_vs_all.shape)
            y_predict = np.array([perceptrons[i].label for i in one_vs_all])
            print(y_predict.shape)
            print("Zero_One_Loss for prediction : ",Z_O_L(YT,y_predict) )
            ZOL.append(Z_O_L(YT,y_predict)) 
            
        print(ZOL)             
        plt.plot([1,2,3,4,5,6,7,8,9,10], ZOL)
        plt.xticks([1,2,3,4,5,6,7,8,9,10])
        plt.ylabel("Zero-One Loss")
        plt.xlabel("Epoch")
        plt.title('Epoch Smallest Error')              
             
test_kernel_degree_avg()
test_kernel_degree_smallest()
test_kernel_epoch_AVG()
test_kernel_epoch_SMERR()