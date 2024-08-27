import pandas as pd

df = pd.read_csv("jobs_dataset_with_features.csv")



# Dropping classes with less than 6500 instances
min_count = 6500
role_counts = df['Role'].value_counts()
dropped_classes = role_counts[role_counts < min_count].index
filtered_df = df[~df['Role'].isin(dropped_classes)].reset_index(drop=True)

# Checking the updated role counts
#print(filtered_df['Role'].value_counts())

df = filtered_df.sample(n=10000)
#print(df)

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

X = df['Features']
y = df['Role']

#train-test-split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2,random_state = 42)

tfidf_vectorizer = TfidfVectorizer()
X_train_idf = tfidf_vectorizer.fit_transform(X_train)
X_test_idf = tfidf_vectorizer.transform(X_test)

#RandomForestClassifier
rf_classifier = RandomForestClassifier()
rf_classifier.fit(X_train_idf,y_train)

#Prediction
y_pred = rf_classifier.predict(X_test_idf)

#Accuracy
accuracy = accuracy_score(y_test,y_pred)
print('Accuracy: ',accuracy)

# Clean resume
import re
def cleanResume(txt):
    cleanText = re.sub('http\S+\s', ' ', txt)
    cleanText = re.sub('RT|cc', ' ', cleanText)
    cleanText = re.sub('#\S+\s', ' ', cleanText)
    cleanText = re.sub('@\S+', '  ', cleanText)  
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText) 
    cleanText = re.sub('\s+', ' ', cleanText)
    return cleanText


# Prediction and Category Name
def job_recommendation(resume_text):
    resume_text = cleanResume(resume_text)
    resume_tfidf = tfidf_vectorizer.transform([resume_text])
    predicted_category = rf_classifier.predict(resume_tfidf)[0]
    return predicted_category



#save models
import pickle
pickle.dump(rf_classifier,open('models/rf_classifier_job_recommendation.pkl','wb'))
pickle.dump(tfidf_vectorizer,open('models/tfidf_vectorizer_job_recommendation.pkl','wb'))