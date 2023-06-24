#!/usr/bin/env python
# coding: utf-8

# In[314]:


import numpy as np
import pandas as pd


# In[315]:


df = pd.read_csv('spam.csv', encoding='latin-1')
df.sample(5)


# In[316]:


df.shape


# In[317]:


# 1. Data cleaning
# 2. EDA
# 3. Text Preprocessing
# 4. Model building
# 5. Evaluation
# 6. Improvement
# 7. Website
# 8. Deploy


# ## 1. Data Cleaning

# In[318]:


df.info()


# In[319]:


# drop last 3 cols
df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)


# In[320]:


df.sample(5)


# In[321]:


# renaming the cols
df.rename(columns={'v1':'target','v2':'text'},inplace=True)
df.sample(5)


# In[322]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()


# In[323]:


df['target'] = encoder.fit_transform(df['target'])


# In[324]:


df.head()


# In[325]:


# missing values
df.isnull().sum()


# In[326]:


# check for duplicate values
df.duplicated().sum()


# In[327]:


# remove duplicates
df = df.drop_duplicates(keep='first')


# In[328]:


df.duplicated().sum()


# In[329]:


df.shape


# ## 2.EDA

# In[330]:


df.head()


# In[331]:


df['target'].value_counts()


# In[332]:


import matplotlib.pyplot as plt
plt.pie(df['target'].value_counts(), labels=['ham','spam'],autopct="%0.2f")
plt.show()


# In[333]:


#data is imbalance


# In[334]:


get_ipython().system('pip install nltk')


# In[335]:


import nltk
nltk.download('punkt')


# In[336]:


df['num_characters'] = df['text'].apply(len)


# In[337]:


df.head()


# In[338]:


# counts the num of words
df['num_words'] = df['text'].apply(lambda x:len(nltk.word_tokenize(x)))


# In[339]:


df.head()


# In[340]:


# counts the num of sentence
df['num_sentences'] = df['text'].apply(lambda x:len(nltk.sent_tokenize(x)))


# In[341]:


df.head()


# In[342]:


df[['num_characters','num_words','num_sentences']].describe()


# In[343]:


# ham
df[df['target'] == 0][['num_characters','num_words','num_sentences']].describe()


# In[344]:


#spam
df[df['target'] == 1][['num_characters','num_words','num_sentences']].describe()


# In[345]:


get_ipython().system('pip install seaborn')
import seaborn as sns


# In[346]:


plt.figure(figsize=(12,6))
sns.histplot(df[df['target'] == 0]['num_characters'])
sns.histplot(df[df['target'] == 1]['num_characters'],color='red')


# In[347]:


plt.figure(figsize=(12,6))
sns.histplot(df[df['target'] == 0]['num_words'])
sns.histplot(df[df['target'] == 1]['num_words'],color='red')


# In[348]:


sns.pairplot(df,hue='target')


# In[349]:


sns.heatmap(df.corr(),annot=True)


# In[350]:


#3. Data Preprocessing
# Lower case
# Tokenization
# Removing special characters
# Removing stop words and punctuation
# Stemming


# In[351]:


import string
string.punctuation


# In[352]:


nltk.download('stopwords')


# In[353]:


import nltk.corpus


# In[354]:


from nltk.corpus import stopwords
stopwords.words('english')


# In[355]:


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
ps.stem('loving')


# In[356]:


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
    
            
    return " ".join(y)


# In[357]:


transform_text("I'm gonna be home soon and i don't want to talk about this stuff anymore tonight, k? I've cried enough today.")


# In[358]:


df['text'][10]


# In[359]:


df['transformed_text'] = df['text'].apply(transform_text)


# In[360]:


df.head()


# In[361]:


from wordcloud import WordCloud
wc = WordCloud(width=500,height=500,min_font_size=10,background_color='white')


# In[362]:


spam_wc = wc.generate(df[df['target'] == 1]['transformed_text'].str.cat(sep=" "))


# In[363]:


plt.figure(figsize=(15,6))
plt.imshow(spam_wc)


# In[364]:


ham_wc = wc.generate(df[df['target'] == 0]['transformed_text'].str.cat(sep=" "))


# In[365]:


plt.figure(figsize=(15,6))
plt.imshow(ham_wc)


# In[366]:


spam_corpus = []
for msg in df[df['target'] == 1]['transformed_text'].tolist():
    for word in msg.split():
        spam_corpus.append(word)
len(spam_corpus)        


# In[367]:


import matplotlib.pyplot as plt
from collections import Counter

spam_counter = Counter(spam_corpus)
most_common = spam_counter.most_common(30)

plot = pd.DataFrame(most_common, columns=['Word', 'Frequency'])

plt.figure(figsize=(12, 6))
sns.barplot(data=plot, x='Word', y='Frequency')
plt.xticks(rotation='vertical')
plt.xlabel('Word')
plt.ylabel('Frequency')
plt.title('Top 30 Words in Spam Corpus')
plt.show()


# In[368]:


# Text Vectorization
# using Bag of Words
df.head()


# ## 4. Model Building

# In[369]:


from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
cv = CountVectorizer()
tfidf = TfidfVectorizer(max_features=3000)


# In[370]:


X = tfidf.fit_transform(df['transformed_text']).toarray()


# In[371]:


X.shape


# In[372]:


y = df['target'].values
y


# In[373]:


from sklearn.model_selection import train_test_split


# In[374]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)


# In[375]:


from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score


# In[376]:


gnb = GaussianNB()
mnb = MultinomialNB()
bnb = BernoulliNB()


# In[377]:


gnb.fit(X_train,y_train)
y_pred1 = gnb.predict(X_test)
print(accuracy_score(y_test,y_pred1))
print(confusion_matrix(y_test,y_pred1))
print(precision_score(y_test,y_pred1))


# In[378]:


mnb.fit(X_train,y_train)
y_pred2 = mnb.predict(X_test)
print(accuracy_score(y_test,y_pred2))
print(confusion_matrix(y_test,y_pred2))
print(precision_score(y_test,y_pred2))


# In[379]:


bnb.fit(X_train,y_train)
y_pred3 = bnb.predict(X_test)
print(accuracy_score(y_test,y_pred3))
print(confusion_matrix(y_test,y_pred3))
print(precision_score(y_test,y_pred3))


# ##### we choose tfidf ---> and  mnb

# In[380]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier


# In[381]:


svc = SVC(kernel='sigmoid', gamma=1.0)
knc = KNeighborsClassifier()
mnb = MultinomialNB()
dtc = DecisionTreeClassifier(max_depth=5)
lrc = LogisticRegression(solver='liblinear', penalty='l1')
rfc = RandomForestClassifier(n_estimators=50, random_state=2)
abc = AdaBoostClassifier(n_estimators=50, random_state=2)
bc = BaggingClassifier(n_estimators=50, random_state=2)
etc = ExtraTreesClassifier(n_estimators=50, random_state=2)
gbdt = GradientBoostingClassifier(n_estimators=50,random_state=2)
xgb = XGBClassifier(n_estimators=50,random_state=2)


# In[382]:


clfs = {
    'SVC' : svc,
    'KN' : knc, 
    'NB': mnb, 
    'DT': dtc, 
    'LR': lrc, 
    'RF': rfc, 
    'AdaBoost': abc, 
    'BgC': bc, 
    'ETC': etc,
    'GBDT':gbdt,
    'xgb':xgb
}


# In[383]:


def train_classifier(clf,X_train,y_train,X_test,y_test):
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred)
    
    return accuracy,precision


# In[384]:


train_classifier(svc,X_train,y_train,X_test,y_test)


# In[385]:


accuracy_scores = []
precision_scores = []

for name,clf in clfs.items():
    
    current_accuracy,current_precision = train_classifier(clf, X_train,y_train,X_test,y_test)
    
    print("For ",name)
    print("Accuracy - ",current_accuracy)
    print("Precision - ",current_precision)
    
    accuracy_scores.append(current_accuracy)
    precision_scores.append(current_precision)


# In[386]:


performance_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy':accuracy_scores,'Precision':precision_scores}).sort_values('Precision',ascending=False)


# In[387]:


performance_df


# In[388]:


performance_df1 = pd.melt(performance_df, id_vars = "Algorithm")


# In[389]:


performance_df1


# In[390]:


sns.catplot(x = 'Algorithm', y='value', 
               hue = 'variable',data=performance_df1, kind='bar',height=5)
plt.ylim(0.5,1.0)
plt.xticks(rotation='vertical')
plt.show()


# In[391]:


# model improve
# 1. Change the max_features parameter of TfIdf


# In[392]:


temp_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy_max_ft_3000':accuracy_scores,'Precision_max_ft_3000':precision_scores}).sort_values('Precision_max_ft_3000',ascending=False)


# In[393]:


temp_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy_scaling':accuracy_scores,'Precision_scaling':precision_scores}).sort_values('Precision_scaling',ascending=False)


# In[394]:


new_df = performance_df.merge(temp_df,on='Algorithm')


# In[395]:


new_df_scaled = new_df.merge(temp_df,on='Algorithm')


# In[396]:


temp_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy_num_chars':accuracy_scores,'Precision_num_chars':precision_scores}).sort_values('Precision_num_chars',ascending=False)


# In[397]:


new_df_scaled.merge(temp_df,on='Algorithm')


# In[398]:


# Voting Classifier
svc = SVC(kernel='sigmoid', gamma=1.0,probability=True)
mnb = MultinomialNB()
etc = ExtraTreesClassifier(n_estimators=50, random_state=2)

from sklearn.ensemble import VotingClassifier


# In[399]:


voting = VotingClassifier(estimators=[('svm', svc), ('nb', mnb), ('et', etc)],voting='soft')


# In[400]:


voting.fit(X_train,y_train)


# In[407]:


mnb.fit(X_train,y_train)


# In[401]:


y_pred = voting.predict(X_test)
print("Accuracy",accuracy_score(y_test,y_pred))
print("Precision",precision_score(y_test,y_pred))


# In[402]:


# Applying stacking
estimators=[('svm', svc), ('nb', mnb), ('et', etc)]
final_estimator=RandomForestClassifier()


# In[403]:


from sklearn.ensemble import StackingClassifier


# In[404]:


clf = StackingClassifier(estimators=estimators, final_estimator=final_estimator)


# In[405]:


clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Accuracy",accuracy_score(y_test,y_pred))
print("Precision",precision_score(y_test,y_pred))


# In[408]:


import pickle
pickle.dump(tfidf,open('vectorizer.pkl','wb'))
pickle.dump(mnb,open('model.pkl','wb'))


# In[ ]:




