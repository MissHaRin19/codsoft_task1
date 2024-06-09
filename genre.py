import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import ConfusionMatrixDisplay
from time import time
from imblearn.over_sampling import RandomOverSampler
import warnings

# Disable warnings
warnings.filterwarnings('ignore')

# Read train data
train_data = pd.read_csv("train_data (1).txt", sep=":::", names=["TITLE", "GENRE", "DESCRIPTION"], engine="python")

# Explore train data
print(train_data.info())
print(train_data.describe(include='object').T)
print(train_data.isnull().sum())
print(train_data.GENRE.unique())

# Read test data
test_data = pd.read_csv("test_data (1).txt", sep=":::", names=["TITLE", "DESCRIPTION"], engine="python")

# Explore test data
print(test_data.info())
print(test_data.describe(include='object').T)
print(test_data.duplicated().sum())

# Visualization
plt.figure(figsize=(10, 10))
sns.countplot(data=train_data, y="GENRE", order=train_data["GENRE"].value_counts().index, palette="YlGnBu")
plt.show()

plt.figure(figsize=(27, 7))
sns.countplot(data=train_data, x="GENRE", order=train_data["GENRE"].value_counts().index, palette="YlGnBu")
plt.show()

# Text preprocessing
nltk.download('stopwords')
nltk.download('punkt')
stemmer = LancasterStemmer()
stop_words = set(stopwords.words("english"))  

def cleaning_data(text):
    text = text.lower()
    text = re.sub(r'@\S+', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'.pic\S+', '', text)
    text = re.sub(r'[^a-zA-Z+]', ' ', text)
    text = "".join([i for i in text if i not in string.punctuation])
    words = nltk.word_tokenize(text)
    text = " ".join([i for i in words if i not in stop_words and len(i) > 2])
    text = re.sub(r"\s+", " ", text).strip()
    return text

train_data["TextCleaning"] = train_data["DESCRIPTION"].apply(cleaning_data)
test_data["TextCleaning"] = test_data["DESCRIPTION"].apply(cleaning_data)

# Feature extraction
vectorize = TfidfVectorizer()
x_train = vectorize.fit_transform(train_data["TextCleaning"])
x_test = vectorize.transform(test_data["TextCleaning"])

# Model training and evaluation
sampler = RandomOverSampler()
x_train_resampled, y_train_resampled = sampler.fit_resample(x_train, train_data['GENRE'])

print('Train:', x_train_resampled.shape[0])
print('Test:', y_train_resampled.shape[0])

y_actual = pd.read_csv("test_data_solution.txt", sep=":::", usecols=[2], header=None).rename(columns={2:'Actual_Genre'})

# Naive Bayes Model
NB = MultinomialNB(alpha=0.3)
start_time = time()
NB.fit(x_train_resampled, y_train_resampled)
y_pred = NB.predict(x_test)
print('Accuracy:', accuracy_score(y_actual, y_pred))
end_time = time()
print('Running Time:', round(end_time - start_time, 2), 'Seconds')

print(classification_report(y_actual, y_pred))

cm = confusion_matrix(y_actual, y_pred, labels=NB.classes_)
cmd = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=NB.classes_)
cmd.plot(cmap=plt.cm.Reds, xticks_rotation='vertical', text_kw={'size': 8})
plt.show()

pd.concat([pd.concat([test_data, y_actual], axis=1), pd.Series(y_pred)], axis=1).rename(columns={0:'Predicted_Genre'}).head(10)

# Modify Genre
y_train_modified = train_data['GENRE'].apply(lambda genre: genre if genre.strip() in ['drama', 'documentary'] else 'other')
y_actual_modified = y_actual['Actual_Genre'].apply(lambda genre: genre if genre.strip() in ['drama', 'documentary'] else 'other')

NB = MultinomialNB(alpha=0.3)
start_time = time()
NB.fit(x_train, y_train_modified)
y_pred = NB.predict(x_test)
print('Accuracy:', accuracy_score(y_actual_modified, y_pred))
end_time = time()
print('Running Time:', round(end_time - start_time, 2), 'Seconds')

# Print Classification Report
print(classification_report(y_actual_modified, y_pred))

# Plot Confusion Matrix
cm = confusion_matrix(y_actual_modified, y_pred, labels=NB.classes_)
cmd = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=NB.classes_)
cmd.plot(cmap=plt.cm.Reds, xticks_rotation='vertical', text_kw={'size': 8})
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
