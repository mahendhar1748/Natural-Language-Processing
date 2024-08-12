# importing the Dataset

import pandas as pd

#Text document---> separated by tab delimeter

messages = pd.read_csv('smsspamcollection/SMSSpamCollection', sep='\t',
                           names=["label", "message"])

#Data cleaning and preprocessing
import re
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
corpus = []
for i in range(0, len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['message'][i])
    review = review.lower()
    review = review.split()
    
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
    
    
# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=2500)  #Taking top 2500 max features(2500 words) 
X = cv.fit_transform(corpus).toarray()


#Converting label column (ham and spam) to the 1 and 0 (categorical variables)  ham column(ham --->1 and spam ---->0),spam column(spam --->1 and ham-->0)
y=pd.get_dummies(messages['label'])
y=y.iloc[:,1].values  #Taking only one column (spam or ham)


# Train Test Split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Training model using Naive bayes classifier  ----> Naive bayes works completley on probability

from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(X_train, y_train)

y_pred=spam_detect_model.predict(X_test)


#Confusion matrx ---> it is normally a 2*2 matrix

from sklearn.metrics import confusion_matrix
confusion_m=confusion_matrix(y_test,y_pred)

#Accuracy score
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)

















