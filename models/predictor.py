import re
import sklearn
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split as splitter
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
import joblib
import tensorflow as tf
import keras
from keras.models import Sequential
from keras import layers
from keras.layers import Dropout
from keras.models import load_model
import statistics as stats



class predictor:
	def __init__(self):
		self.nb= None
		self.svm=None
		self.lr=None
		self.vec=None
		self.ann=None

	def preprocessing(self,data):
		processed_data=[]
		for i in data:
			x = re.sub(r'[^a-z]+',' ',i)
			processed_data.append(x)

		return processed_data

	def load_data(self,filename):
		lyrics = pd.read_csv('data.csv')
		lyrics_data=list(lyrics['lyrics'].str.lower())
		artist_data=list(lyrics['artist'])

		return lyrics_data,artist_data

	def split(self,data,labels):
		train_data,test_data,train_labels,test_labels=splitter(data,labels,train_size=0.8,test_size=0.2,shuffle=True)
		return train_data,train_labels,test_data,test_labels

	def encoded(self,labels):
		label_encoder = LabelEncoder()
		integer_encoded = label_encoder.fit_transform(labels)

		return np.array(integer_encoded)

	def onehot(self,integer_encoded):
		onehot_encoder = OneHotEncoder(sparse=False)
		integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
		onehot_encoded= onehot_encoder.fit_transform(integer_encoded)
		return np.array(onehot_encoded)

	def vectorizer(self,data):
		vectorizer = CountVectorizer()
		return vectorizer.fit(data)

	def logistic(self,matrix,labels):
		classifier = LogisticRegression(max_iter=1000)
		classifier.fit(matrix, labels)
		return classifier

	def SVM(self,matrix,labels):
		Linear_SVM = svm.LinearSVC()
		Linear_SVM.fit(matrix,labels)
		return Linear_SVM

	def naive(self,matrix,labels):
		Naive2 = MultinomialNB()
		Naive2.fit(matrix.toarray(),labels)
		return Naive2

	def train(self):
		data,labels = self.load_data('data.csv')
		processed_data = self.preprocessing(data)
		train_data,train_labels,test_data,test_labels=self.split(processed_data,labels)
		encoded_train_labels= self.encoded(train_labels)
		encoded_test_labels = self.encoded(test_labels)
		vectorizer = self.vectorizer(train_data)
		joblib.dump(vectorizer,'vec.sav')
		train_vectors = vectorizer.transform(train_data)
		test_vectors = vectorizer.transform(test_data)

		#training
		naive = self.naive(train_vectors,encoded_train_labels)
		print(naive.score(test_vectors,encoded_test_labels))

		logistic = self.logistic(train_vectors,encoded_train_labels)
		print(logistic.score(test_vectors,encoded_test_labels))

		SVM = self.SVM(train_vectors,encoded_train_labels)
		print(SVM.score(test_vectors,encoded_test_labels))


		self.nb=naive
		self.lr=logistic
		self.svm=SVM
		self.save()

	def save(self):
		joblib.dump(self.nb,'nb.sav')
		joblib.dump(self.lr,'lr.sav')
		joblib.dump(self.svm,'svm.sav')
		print('saved')

	def load(self):
		self.lr = joblib.load('./models/lr.sav')
		self.svm = joblib.load('./models/svm.sav')
		self.nb=joblib.load('./models/nb.sav')
		self.vec = joblib.load('./models/vec.sav')
		

	def predict(self,input):
		self.load()
		processed = self.preprocessing([input])
		vec=self.vec.transform(processed)
		predictions=[]
		artist={0:"Eminem",1:"G-Eazy",2:"Kendrick Lamar",3:"Machine Gun Kelly"}
		models=["Naive Bayes","Logistic Regression","Support Vector Machine"]
		predictions.append((self.nb.predict(vec)[0],self.nb.predict_proba(vec)))
		predictions.append((self.lr.predict(vec)[0],self.lr.predict_proba(vec)))
		predictions.append((self.svm.predict(vec)[0],0))
		print(predictions)
		results=[]
		for index,i in enumerate(predictions):
			results.append({
				"Model":models[index],
				"Artist":artist[i[0]],
				"Proba":i[1] 
				})

		return results
		
	def art(self):

		data,labels = self.load_data('data.csv')
		processed_data = self.preprocessing(data)
		train_data,train_labels,test_data,test_labels=self.split(pocessed_data,labels)
		encoded_train_labels= self.encoded(train_labels)
		encoded_test_labels = self.encoded(test_labels)
		vectorizer = self.vectorizer(train_data)
		train_vectors = vectorizer.transform(train_data)
		test_vectors = vectorizer.transform(test_data)
		model = Sequential()
		model.add(layers.Dense(100,input_dim=train_vectors.shape[1],activation='sigmoid'))
		model.add(Dropout(0.5))
		model.add(layers.Dense(100,activation='sigmoid'))
		model.add(Dropout(0.5))
		model.add(layers.Dense(4,activation='relu'))
		model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
		print(model.summary())
		one_encoded_train=self.onehot(encoded_train_labels)
		model.fit(train_vectors.toarray(),one_encoded_train, epochs=80, batch_size=50,verbose=1,shuffle=True)
		return model

	







