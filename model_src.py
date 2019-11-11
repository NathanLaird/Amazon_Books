import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
import nltk
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
nltk.download('wordnet')
from sklearn.svm import SVC
from xgboost import XGBClassifier

import tensorflow as tf
import tensorflow_hub as hub
import pickle


class NLP_model:
	def __init__(self,model_name='SVC',params={}):
		self.model_name = model_name
		if model_name =='SVC':
			self.model = svm.SVC(C=1.0, kernel='linear', degree=6, gamma='auto')
			
			
		if model_name =='Naive_Bayes':
			self.model = naive_bayes.MultinomialNB()
			
		if model_name =='RF':
			self.model = RandomForestClassifier(n_estimators=params['n_estimators'])
		if model_name == 'LR':
			self.model = LogisticRegression()
		if model_name == 'XGBoost':
			self.model = XGBClassifier(max_depth=10, n_estimators=1000, learning_rate=0.01)
		if model_name =='Bert':
			print('Berts')


	def fit(self,Train_X,Train_Y):
		self.model.fit(Train_X,Train_Y)

	def predict(self,Test_X):
		return self.model.predict(Test_X)

	def predict_proba(self,Test_X):
		return self.model.predict_proba(Test_X)







"""
def model_fit(Train_X,Train_Y,model_name='SVC'):
	if model_name =='SVC':
		SVM = svm.SVC(C=1.0, kernel='linear', degree=6, gamma='auto')
		SVM.fit(Train_X,Train_Y)
		return SVM
	if model_name =='Naive_Bayes':
		NB = naive_bayes.MultinomialNB()
		NB.fit(Train_X,Train_Y)
		return NB
	if model_name =='RF':
		print('rf')
"""

