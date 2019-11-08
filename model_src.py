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
nltk.download('wordnet')
from sklearn.svm import SVC



def model_fit(Train_X,Train_Y,model_name='SVC'):
	if model_name =='SVC':
		SVM = svm.SVC(C=1.0, kernel='linear', degree=6, gamma='auto')
		SVM.fit(Train_X,Train_Y)
		return SVM