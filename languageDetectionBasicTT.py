# -*- coding: utf-8 -*-
"""
Created on Sat May 19 19:10:51 2018

@author: Mustafa
"""
import numpy
import pandas
import random
import pandas as pd

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.decomposition import PCA as sklearnPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.datasets.samples_generator import make_blobs
from keras import models
from keras import layers
from pandas.tools.plotting import parallel_coordinates
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import r2_score
#Yüklenecek olan datasetlerin uzantısı
xTrainFilePath = "C:\\Users\\Mustafa\\Desktop\\dersler\\bill443Project\\wili-2018\\x_train.txt"
yTrainFilePath = "C:\\Users\\Mustafa\\Desktop\\dersler\\bill443Project\\wili-2018\\y_train.txt"

stateOfCoding = False

#Burada her dile ait alfabeler bir listeye kaydediliyor
turAl = list("abcçdefgğhıijklmnoöprsştuüvyz")
engAl = list("abcdefghijklmnopqrstuvwxyz")
espAl = list("abcdefghijklmnñopqrstuvwxyz")
gerAl = list("abcdefghijklmnopqrstuvwxyzäöüß")
freAl = list("abcdefghijklmnopqrstuvwxyz")
itaAl = list("abcdefghilmnopqrstuv")
bulAl = list("абвгдежзийклмнопрстуфхцчшщъьюя")
czeAl = list("aábcčdďeéěfghiíjklmnňoóprřsštťuúůvyýzž")
sloAl = list("aáäbcčdďeéfghiíjklĺľmnňoóôpqrŕsštťuúvwxyýzž")
#Feature vektörü olarak kullanılması planlanan liste 
merged_list = list("abcdefghiklmnoprstuv")
#merged_list = sorted(list(set(engAl+turAl+espAl+gerAl+freAl+itaAl)))
#merged_list = sorted(list(set(czeAl+sloAl)))
#merged_list = sorted(list(set(engAl+turAl+espAl+gerAl+freAl+itaAl+czeAl+sloAl)))
#merged_list = list("aábcčdďeéfghiíjklmnňoóprsštťuúvyýzž")
#Eğitim verileri ve hangi dile ait olduklarının okutulduğu satır
import codecs
with codecs.open(xTrainFilePath,'r',encoding='utf8') as f:
    x_dataset = f.readlines()
with codecs.open(yTrainFilePath,encoding='utf8') as f:
    y_dataset = f.readlines()
#Selected language hangi dil seçeneklerini eğiteceğimizi belirten dizidir.
#Buraya girilecek olan indisler y_train'den alınmalıdır
#selectedLanguage = ["tur\n","spa\n","eng\n","fra\n","deu\n","ita\n"]
#selectedLanguage = ["ces\n","slk\n"]
selectedLanguage = ["tur\n","spa\n","eng\n","fra\n","deu\n","ita\n","ces\n","slk\n"]
#Elimizdeki datasette bir sürü dil bulunduğundan dolayı böyle bir tanımlama yapmak zorunda kaldık
#Indice matrisi seçilen dillerin verilerin bulunduğu indisleri döndürmektedir.
indices = []
import shutil
import pickle
if stateOfCoding:
    for j in range(0,len(selectedLanguage)):
        indices.extend([i for i, x in enumerate(y_dataset) if x == selectedLanguage[j]])
    #Bulunan indisler belli bir düzenle geldiği seçimi randomlaştırdık.
    #Örneğin ilk yüz verimiz ingilizce sonraki yüz türkçe biz rastgele atınca bu 200 veri karışık dağılıyor. 
    random.shuffle(indices)
    #two ways to delete file
    #shutil.os.remove('EightLanguageIndex.file')
    #write to output
    with open('EightLanguageIndex.file', 'ab') as f:
        for array in indices:
            pickle.dump(array, f)
else:
    #read to input
    with open('EightLanguageIndex.file', 'rb') as f:
        while True:
            try:
                indices.append(pickle.load(f))
            except EOFError:
                break
x_train = []
y_train = []
#Datasetlerdeki satırlardan yukarıdaki indice matrisi yardımıyla doğru verileri ve sonuçlarını çekiyoruz
for i in range(0,len(indices)):
    x_train.append(x_dataset[indices[i]])
    y_train.append(y_dataset[indices[i]])
#Bir sentence içerisinde hangi karaketerden kaç tane olduğunu hesaplayan kod parçası 
def histogramFunc(sentence,charlist):
    countingList = []
    for i in range(0,len(charlist)):
        countingList.append(sentence.count(charlist[i]))
    return countingList
y_train_numericalValue = y_train;
#değiştirilmeli
for i in range(0,len(y_train)):
    for j in range(0,len(selectedLanguage)):
        if(y_train[i] == selectedLanguage[j]):
            y_train[i] = j
import pandas as pd
import numpy as np
# Convert feature matrix into DataFrame

#String outputları rakamlara dönüştüren kod parçası
# encode class values as integers
encoder = LabelEncoder()
encoder.fit(y_train)
encoded_Y = encoder.transform(y_train)
# convert integers to dummy variables (i.e. one hot encoded)
y_notencoding = y_train;
y_train = np_utils.to_categorical(y_train)

template_train = numpy.zeros((len(merged_list),len(indices)))
#Elde edilen sayımlar normalize ediliyor
for i in range(0,len(x_train)):
    countingList = histogramFunc(x_train[i],merged_list)
    x_train[i] = numpy.true_divide(countingList,sum(countingList))

for i in range(0,len(x_train)):
   for j in range(0,len(x_train[i])):
       template_train[j][i] = x_train[i][j]

x_train = template_train
x_train = x_train.transpose()
classifier = Sequential()

filepath='/Users/Mustafa/Desktop/mc/weights-improvement-{epoch:02d}-{loss:.4f}.hdf5'
#pca = sklearnPCA(n_components=2) #2-dimensional PCA


y_train_numericalValue = pd.Series(y_train_numericalValue)
colorLabel = ['green','blue','red','black','orange','purple','yellow','pink']

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot
#3D görünüm için gerekli komutlar ax ile çalıştırılımaktadır.
#fig = pyplot.figure()
#ax = Axes3D(fig)
#for i in range(0,len(selectedLanguage)):
    #plt.scatter(lda_transformed[y_train_numericalValue==i][0], lda_transformed[y_train_numericalValue==i][1], label=selectedLanguage[i], c=colorLabel[i])
    #ax.scatter(lda_transformed[y_train_numericalValue==i][0], lda_transformed[y_train_numericalValue==i][1],lda_transformed[y_train_numericalValue==i][2], label=selectedLanguage[i], c=colorLabel[i])
#plt.legend()
#plt.show()
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
#test datası
# Adding the input layer and the first hidden layer


# Create function returning a compiled network
def create_network():
    
    # Start neural network
    network = models.Sequential()

    # Add fully connected layer with a ReLU activation function
    network.add(layers.Dense(units=30, activation='relu', input_shape=(len(merged_list),)))

    # Add fully connected layer with a ReLU activation function
    network.add(layers.Dense(units=30, activation='relu'))

    # Add fully connected layer with a sigmoid activation function
    network.add(layers.Dense(units=len(selectedLanguage), activation='sigmoid'))

    # Compile neural network
    network.compile(loss='binary_crossentropy', # Cross-entropy
                    optimizer='rmsprop', # Root Mean Square Propagation
                    metrics=['accuracy']) # Accuracy performance metric
    
    # Return compiled network
    return network
# Wrap Keras model so it can be used by scikit-learn
neural_network = KerasClassifier(build_fn=create_network, 
                                 epochs=100, 
                                 batch_size=100, 
                                 verbose=0)
# Evaluate neural network using three-fold cross-validation
score = cross_val_score(neural_network, x_train, y_train, cv=6)
score = score.mean()
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

# Compiling the ANN
filename = "C:\\Users\\Mustafa\\Desktop\\mc\\eightLanguageAlphabetCommonLetter.hdf5"
#classifier.load_weights(filename)
# Fitting the ANN to the Training set
neural_network.fit(x_train, y_train, batch_size = 20, nb_epoch = 100,callbacks=callbacks_list)
#Confusion matris hesaplaması
y_pred = cross_val_predict(neural_network,x_train,y_train,cv=6)
conf_mat = confusion_matrix(y_train_numericalValue,y_pred)
#accuracies.std() 
#sentence = "Dnešné protestné zhromaždenia sú v poradí už ôsme od vraždy investigatívneho novinára Aktuality.sk Jána Kuciaka a jeho snúbenice Martiny Kušnírovej"
sentence = "Unsere Hörtexte werden von Journalisten und Experten für Deutsch als Fremdsprache konzipiert und von Muttersprachlern gelesen. So sind die Inhalte authentisch und gleichzeitig für Deutschlerner gut verständlich. Hier finden Sie interessante Hörartikel und praktische Übungen – zum Beispiel zu Grammatik oder Redewendungen – in drei Sprachniveaus. Für jeden Text gibt es die Audiodatei zum Anhören und ein Transkript zum Mitlesen."
sentence = sentence.lower()
test = numpy.zeros((2,len(merged_list)))
countingList = numpy.zeros((2,len(merged_list)))
countingList[0,:] = histogramFunc(sentence,merged_list)

from sklearn.metrics import accuracy_score
test[0,:] = numpy.true_divide(countingList[0,:],sum(countingList[0,:]))
#y_pred = neural_network.predict(x_train[3201:4000,:])
#y_pred = neural_network.predict(test[0:1,:])

#y_pred = numpy.around(y_pred, 0)
#score = accuracy_score(y_train[3201:4000], y_pred)


    




    
    