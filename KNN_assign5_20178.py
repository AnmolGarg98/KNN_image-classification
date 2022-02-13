from keras.applications.vgg16 import preprocess_input 
from keras.preprocessing.image import ImageDataGenerator

# models 
from keras.applications.vgg16 import VGG16 
from keras.models import Model

# clustering and dimension reduction
# from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# for everything else
import numpy as np
# import pandas as pd
import pickle

datagen = ImageDataGenerator()
"""
    # path to DataGen folder
    # DataGen folder must contain two folders inside with name test and train
      with each folder containing folders having different image types
    # DataGen/train -->airplanes,bikes,cars,faces folders
    # DataGen/test -->airplanes,bikes,cars,faces folders
"""
home_path = r'D:\sem1_2021\DIP\assinments\Assignment05\Images\DataGen'

print("getting data using ImageDataGenerator")
train_data    =    datagen.flow_from_directory(
                  directory=home_path + r'/train/',
                  target_size=(224,224), # resize to this size to the size required fo VGG16
                  color_mode="rgb", # for coloured images
                  batch_size=1, # number of images to extract from folder for every batch
                  class_mode="binary", # classes to predict (single class classifier)
                  )

test_data    =    datagen.flow_from_directory(
                  directory=home_path + r'/test/',
                  target_size=(224,224), # resize to this size to the size required fo VGG16
                  color_mode="rgb", # for coloured images
                  batch_size=1, # number of images to extract from folder for every batch
                  class_mode="binary", 
                  )

            
model = VGG16()
model = Model(inputs = model.inputs, outputs = model.layers[-2].output) #taking features from the secondlast layer of VGG16


def extract_features(file, model):
    imgx = preprocess_input(file) #reshaped_img
    # get the feature vector
    features = model.predict(imgx, use_multiprocessing=True)
    return features
   
data = {}
p = r'D:\sem1_2021\DIP\assinments\Assignment05\Images\except'

print("exracting features of train/test image using VGG")
features_train = [] #array containg features of each image
labels_train = []   #array containg label(class of img)
i=0
for i in range(120): # 120 is number of traing images
    print("train" ,i)
    
    # extract the features and update the dictionary
    batchX, batchY = train_data.next() # batchx contains the image aray of particular index
    try:                               #  batchy contains the label number present in train_data from DataGen operation
        feat = extract_features(batchX,model) #getting features of particular image from VGG model
        labels_train.append(batchY)
        features_train.append(feat) 
    # error handling / can ignore
    except:
        with open(p,'wb') as file:
            pickle.dump(data,file)

# similar as train_data operation
features_test = []   
labels_test = []
i=0
for i in range(80):
    print("test",i)
    # try to extract the features and update the dictionary
    batchX, batchY = test_data.next()
    try:
        feat = extract_features(batchX,model)
        labels_test.append(batchY)
        features_test.append(feat) 
    # if something fails, save the extracted features as a pickle file (optional)
    except:
        with open(p,'wb') as file:
            pickle.dump(data,file)


features_train = np.array(features_train)
labels_train = np.array(labels_train)
features_test = np.array(features_test)
labels_test = np.array(labels_test)
# reshape so that there are 120 and 80 respective samples of 4096 vectors
features_train = features_train.reshape(-1,4096)
# print(features_train.shape)
features_test = features_test.reshape(-1,4096)

# reduce the amount of dimensions in the feature vector by extracting most dependent featues only using PCA
print("PCA_TRAIN")
pca = PCA(n_components=40, random_state=78)  #4096 to 40 features for easy computation by our KNN
pca.fit(features_train)
x_train = pca.transform(features_train)

print("PCA_TEST")
pca = PCA(n_components=40, random_state=78)
pca.fit(features_test)
x_test = pca.transform(features_test)

print("KNN_MODEL")

training_data = np.column_stack((x_train,labels_train)) #merging the two arrays to one to pass to KNN function
testing_data = np.column_stack((x_test,labels_test))

def EUC_DIST(v1,v2): #function returning euclidean distance between any two vectors of equal dim
    v1,v2 = np.array(v1),np.array(v2)
    
    distance = 0
    
    for i in range(len(v1)-1):
        distance += (v1[i]-v2[i])**2
        
    return np.sqrt(distance)

def Predict(k,train_data,test_instance): # k = number of nearest neighb ,train_data = whole train array , test = only one single test image and its label
    distances = [] #array containing euc dist of test image with every training image respectively
    
    for i in range(len(train_data)):
        dist = EUC_DIST(train_data[i][:-1], test_instance)
        distances.append((train_data[i],dist)) 
    
    distances.sort(key=lambda x: x[1]) #sorting with least distance on top
    
    neighbors = []
    for i in range(k):
        neighbors.append(distances[i][0]) #contain array of labels of image with least euc dist to test image
        
    classes = {}
    for i in range(len(neighbors)):
        response = neighbors[i][-1]
        
        if response in classes:
            classes[response] += 1
            
        else:
            classes[response] = 1
    
    sorted_classes = sorted(classes.items() , key = lambda x: x[1],reverse = True )
    
    return sorted_classes[0][0]  #return the predicted class/label of test img

def Eval_Acc(y_data,y_pred): #function to calculate accuracy from 80 predicted images
    correct = 0
    
    for i in range(len(y_pred)):
        if y_data[i][-1] == y_pred[i]:  #if given data image label is equal to prdicted label of test image
            correct += 1
    return (correct / len(y_pred))*100
    
y_pred = [] #array containg KNN predicted labels/class of each image in test_data
for i in range(len(testing_data)):
    y_pred.append(Predict(2,training_data, testing_data[i]))
  
print(Eval_Acc(testing_data, y_pred))
     
    
    
    
    
    
    