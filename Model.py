import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import re
import nltk
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


data='D:\Resume_classification\Project\Resume.csv'
df=pd.read_csv(data)
df.info()

dropped_columns=['ID', 'Resume_html']
df=df.drop(dropped_columns, axis=1)
print(df)

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

col_to_clean=['Resume_str']


def clean_data(text):
    text = text.lower()  # Converting all letters to lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Removing punctuations
    text = re.sub(r'<.*?>', '', text)  # Removing HTML tags
    text = re.sub(r'https\S+\s*', '', text)  # Removing http/urls
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Removing non-English letters

    text = re.sub(r'#\S+', '', text)  # Removing hashtags
    text = re.sub(r'@\S+', '', text)  # Removing mentions
    text = re.sub(r'\s+', ' ', text)  # Removing extra spaces
    text = re.sub(r'(.)\1+', r'\1', text)  # Removing repeated characters

    text=nltk.tokenize.word_tokenize(text) #Tokenizing words

    #text = [w for w in text if not w in nltk.corpus.stopwords.words('english')]

    stop_words=set(stopwords.words('english'))
    filtered_words = [w for w in text if w not in stop_words] #removing stopwords

    return ' '.join(filtered_words)

cleaned_df=df.copy()
cleaned_df[col_to_clean]=cleaned_df[col_to_clean].applymap(clean_data)


print(cleaned_df)

from sklearn.preprocessing import LabelEncoder

var=['Category']
le=LabelEncoder()
for i in var:
  cleaned_df[i]=le.fit_transform(cleaned_df[i])

cleaned_df


categories = np.sort(cleaned_df['Category'].unique())
len(categories)
from sklearn.model_selection import train_test_split

# Splitting the dataset into train and temporary (test + validation) sets
train_data, temp_data = train_test_split(cleaned_df, test_size=0.3, random_state=42)

# Splitting the temporary set into test and validation sets
test_data, validation_data = train_test_split(temp_data, test_size=0.5, random_state=42)

# Print the sizes of the resulting sets
print("Train set size:", len(train_data))
print("Test set size:", len(test_data))
print("Validation set size:", len(validation_data))

from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

train_text=train_data['Resume_str'].values
train_target =train_data['Category'].values

word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    stop_words='english',max_features=2000)
word_vectorizer.fit(train_text)
WordFeatures = word_vectorizer.transform(train_text)

WordFeatures

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(WordFeatures,train_target, random_state=1, test_size=0.2,
                                                 stratify=train_target)
print(X_train.shape)
print(X_test.shape)

from sklearn.metrics import accuracy_score
from sklearn import metrics

from sklearn import svm
model_svm = svm.SVC()
model_svm.fit(X_train, y_train) #training the model with the training dataset
y_prediction_svm = model_svm.predict(X_test) #

# by comparing predicted output by the model and the actual output
score_svm = metrics.accuracy_score(y_prediction_svm, y_test).round(4)
print("----------------------------------")
print('The accuracy of the SVM is: {}'.format(score_svm))
print("----------------------------------")
# saving the accuracy score
score = set()
score.add(('SVM', score_svm))

from sklearn.tree import DecisionTreeClassifier
model_dt = DecisionTreeClassifier(random_state=4)
model_dt.fit(X_train, y_train)
y_prediction_dt = model_dt.predict(X_test)

score_dt = metrics.accuracy_score(y_prediction_dt, y_test).round(4)
print("---------------------------------")
print('The accuracy of the DT is: {}'.format(score_dt))
print("---------------------------------")

score.add(('DT', score_dt))

from sklearn.linear_model import LogisticRegression
model_lr = LogisticRegression()
model_lr.fit(X_train, y_train)
y_prediction_lr = model_lr.predict(X_test)


score_lr = metrics.accuracy_score(y_prediction_lr, y_test).round(4)
print("---------------------------------")
print('The accuracy of the LR is: {}'.format(score_lr))
print("---------------------------------")
# save the accuracy score
score.add(('LR', score_lr))


from sklearn.metrics import classification_report

models = [model_svm, model_lr, model_dt]

for model in models:
    y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred)

print("Model:", model)
print(report)
print("-" * 50)

model_filename = 'model_svm.pkl'
joblib.dump(model_svm, model_filename)
print(f"Trained SVM model saved as '{model_filename}'")