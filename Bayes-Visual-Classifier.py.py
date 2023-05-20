import pickle
import numpy as np
import matplotlib.pyplot as plt
from random import random
from skimage.transform import resize
from scipy.stats import norm
from scipy.stats import multivariate_normal

def  class_acc(pred,gt):
    accuracy=1
    for i,ivalue in enumerate(gt):
        if pred[i]!=gt[i]:
            accuracy=accuracy-(1/len(gt))
    return(accuracy)

#Bayesian Classifier

def cifar_10_bayes_learn(X,Y):
    X_c=[]
    X_m=[]
    X_cov=[]
    P_c=[]
    X_c=np.array([[X[i] for i,ivalue in enumerate(Y) if Y[i]==j] for j in range(10)])
    X_m=np.array([[np.mean(X_c[j,0:,i]) for i in range(X_c.shape[2])] for j in range(10)])
    X_cov=np.array([np.cov(X_c[i,:,:].reshape(5000,X_c.shape[2]), rowvar=False) for i,ivalue in enumerate(X_c)])
    P_c=np.array([(len(X_c[i])/50000) for i in range(10)])
    return X_c,X_m,X_cov,P_c

def cifar10_classifier_bayes(x,mu,sigma,p):
    y_pred1=[]
    for i,ivalue in enumerate(x):
        temp1=[(multivariate_normal.pdf(x[i],mu[j],sigma[j])*p[j]) for j,jvalue in enumerate(mu)]
        yindex1=np.argmax(temp1)
        y_pred1=np.append(y_pred1,yindex1)
    return y_pred1

# Naive Bayes

# Resize Fucntion for 1x1
def cifar10_color(X):
    X=resize(X, (50000,1,1,3),preserve_range=True)
    return X

# Resize Fucntion for 2x2
def cifar10_2x2_color(X):
    X=resize(X, (50000,2,2,3),preserve_range=True)
    return X

def cifar_10_naivebayes_learn(X,Y):
    X_class=[]
    mean_tr=[]
    std_tr=[]
    p_class=[]
    X_class=np.array([[X[i] for i,ivalue in enumerate(Y) if Y[i]==j] for j in range (10)])
    mean_tr=np.array([[np.mean(X_class[j,0:,:,:,i]) for i in range(3)] for j,jvalue in enumerate(X_class)])
    std_tr=np.array([[np.std(X_class[j,0:,:,:,i]) for i in range(3)] for j,jvalue in enumerate(X_class)])
    p_class=np.array([(len(X_class[i])/50000) for i in range(10)])
    return X_class,mean_tr,std_tr,p_class

def cifar10_classifier_naivebayes(x,mu,sigma,p):
    y_pred=[]
    for i,ivalue in enumerate(x):
        temp=[np.prod(norm.pdf(x[i],mu[j],sigma[j])*p[j]) for j,jvalue in enumerate(mu)]
        yindex=np.argmax(temp)
        y_pred=np.append(y_pred,yindex)
    return y_pred

def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict

batchsize= 5
X=np.ndarray((0,3072))
Y=np.ndarray(0).astype('int')

for l in range(1,batchsize+1):
    datadict= unpickle('C:/Users/Muhammad Shahzaib/OneDrive - TUNI.fi/Desktop/Student Docs/Tampere University/Period 1/Intro to Pattern Recognition and Machine Learning/Week 3/cifar-10-batches-py/data_batch_' + str(l))
    X = np.concatenate((X,datadict["data"]))
    Y = np.concatenate((Y,datadict["labels"]))

datadict_test = unpickle('C:/Users/Muhammad Shahzaib/OneDrive - TUNI.fi/Desktop/Student Docs/Tampere University/Period 1/Intro to Pattern Recognition and Machine Learning/Week 3/cifar-10-batches-py/test_batch')


X_te = datadict_test["data"]
Y_te = datadict_test["labels"]

print(X.shape)



labeldict = unpickle('C:/Users/Muhammad Shahzaib/OneDrive - TUNI.fi/Desktop/Student Docs/Tampere University/Period 1/Intro to Pattern Recognition and Machine Learning/Week 3/cifar-10-batches-py/batches.meta')
label_names = labeldict["label_names"]

X = X.reshape(50000, 3, 32, 32).transpose(0,2,3,1).astype("uint8").astype("int")
Y = np.array(Y)

X_te = X_te.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8").astype("int")
Y_te = np.array(Y_te)
X_te_resized = resize(X_te,(10000,1,1,3))

for i in range(X.shape[0]):
      # Show some images randomly
      if random() > 0.999999:
          plt.figure(1);
          plt.clf()
          plt.imshow(X[i])
          plt.title(f"Image {i} label={label_names[Y[i]]} (num {Y[i]})")
          plt.pause(1)
         
#Main

X_resized = cifar10_color(X)

#Naive Bayes, Question 1:

X_class,mean_tr,std_tr,p_class = cifar_10_naivebayes_learn(X_resized,Y)
y_pred=cifar10_classifier_naivebayes(X_te_resized,mean_tr,std_tr,p_class)
print("The Naive Bayes accuracy = ",class_acc(y_pred,Y_te)*100, "%")

#Multivariate Bayesian Classifier with different image shapes, Question 2 and 3:
imagesize=[1,2,4,8,16]
accuracyplot=[]
for l in range(5):
    k=2**l
    X_resized_new= resize(X,(50000,k,k,3),preserve_range=True)
    X_resized_new= X_resized_new.reshape(50000,k*k*3)
    X_te_2x2 = resize(X_te,(10000,k,k,3),preserve_range=True).reshape(10000,k*k*3)
    X_c,X_m2x2,X_cov2x2,P_c2x2=cifar_10_bayes_learn(X_resized_new,Y)
    y_p2x2=cifar10_classifier_bayes(X_te_2x2,X_m2x2,X_cov2x2,P_c2x2)
    accuracyplot=np.append(accuracyplot,(class_acc(y_p2x2,Y_te)*100))
    print(f'The Bayes accuracy for shape {k}x{k} = {class_acc(y_p2x2,Y_te)*100} %')
plt.clf()
plt.plot(imagesize,accuracyplot, label='Bayes')
plt.xlabel("Image size")
plt.ylabel("Accuracy in %")