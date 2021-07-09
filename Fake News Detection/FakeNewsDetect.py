from sklearn.naive_bayes import MultinomialNB  

from sklearn.linear_model import LogisticRegression  

from sklearn.linear_model import PassiveAggressiveClassifier  

from sklearn.model_selection import train_test_split  

from sklearn.feature_extraction.text import TfidfVectorizer  

from sklearn.feature_extraction.text import CountVectorizer  

from sklearn import metrics  

import itertools  

import matplotlib.pyplot as plt 

train = pd.read_csv("../fake-news/train.csv")  

test  = pd.read_csv ("../fake-news/test.csv")  

  

train.isnull().sum()  

train.dtypes.value_counts()  

test=test.fillna(' ')  

train=train.fillna(' ')  

test['total']=test['title']+' '+test['author']+' '+test['text']train['total']=train['title']+' '+train['author']+' '+train['text']  

train.info()  

train.head()  

X_train, X_test, y_train, y_test = train_test_split(train['total'], train.label, test_size=0.20, random_state=0) 

count_vectorizer = CountVectorizer(ngram_range=(1, 2), stop_words='english')  

count_train = count_vectorizer.fit_transform(X_train)  

count_test = count_vectorizer.transform(X_test) 

tfidf_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))tfidf_train = tfidf_vectorizer.fit_transform(X_train) 

tfidf_test = tfidf_vectorizer.transform(X_test) 

def  plot_confusion_matrix(cm, classes, normalize=False,title='Confusion matrix', cmap=plt.cm.Blues):  

          plt.imshow(cm, interpolation='nearest', cmap=cmap)  

          plt.title(title)  

          plt.colorbar()  

          tick_marks = np.arange(len(classes))      

          plt.xticks(tick_marks, classes, rotation=45)             

          plt.yticks(tick_marks, classes)    thresh = cm.max()   

         fori,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):  

                 plt.text(j, i, cm[i, j],  

                 horizontalalignment"center",  

                 color="white" if cm[i, j] > thresh else "black")      

        plt.tight_layout()      

        plt.ylabel('True label')      

        plt.xlabel('Predicted label') 

nb_classifier = MultinomialNB(alpha = 0.1) 

nb_classifier.fit(count_train, y_train)  

pred_nb_count = nb_classifier.predict(count_test)  

acc_nb_count = metrics.accuracy_score(y_test, pred_nb_count)print(acc_nb_count)  

#Tuning the model 

  for alpha in np.arange(0,1,.05):  

     nb_classifier_tune = MultinomialNB(alpha=alpha)  

     nb_classifier_tune.fit(count_train, y_train)     

     pred_tune = nb_classifier_tune.predict(count_test)  

     score = metrics.accuracy_score(y_test, pred_tune)      

     print("Alpha: {:.2f} Score: {:.5f}".format(alpha, score))   

 

     nb_classifier = MultinomialNB(alpha = 0.15) 

     nb_classifier.fit(count_train, y_train)  

pred_nb_count = nb_classifier.predict(count_test)  

cm = metrics.confusion_matrix(y_test, pred_nb_count, labels=[0,1])  

 

        plot_confusion_matrix(cm, classes=['TRUE','FAKE'], title ='Confusion matrix for a MultinomialNB with Count Vectorizer')    
		
nb_classifier = MultinomialNB(alpha = 0.1) 

nb_classifier.fit(tfidf_train, y_train) 

pred_nb_tfidf = nb_classifier.predict(tfidf_test) 

acc_nb_tfidf = metrics.accuracy_score(y_test, pred_nb_tfidf) 

print(acc_nb_tfidf) 

 

#Tuning the model 

for alpha in np.arange(0,0.1,.01): 

         nb_classifier_tune= MultinomialNB(alpha=alpha) 

         nb_classifier_tune.fit(tfidf_train, y_train) 

         pred_tune = nb_classifier_tune.predict(tfidf_test) 

         score = metrics.accuracy_score(y_test, pred_tune) 

         print("Alpha: {:.2f}  Score: {:.5f}".format(alpha, score)) 

 

nb_classifier = MultinomialNB(alpha = 0.01) 

nb_classifier.fit(tfidf_train, y_train) 

pred_nb_tfidf = nb_classifier.predict(tfidf_test) 

cm2 = metrics.confusion_matrix(y_test, pred_nb_tfidf, labels=[0,1])plot_confusion_matrix(cm2, classes=['TRUE','FAKE'], title ='Confusion matrix for a MultinomialNB with Tf-IDF') 

		
logreg = LogisticRegression(C=1e5)  

logreg.fit(tfidf_train, y_train)  

pred_logreg_tfidf = logreg.predict(tfidf_test)  

pred_logreg_tfidf_proba = logreg.predict_proba(tfidf_test)[:,1]  

acc_logreg_tfidf = metrics.accuracy_score(y_test,pred_logreg_tfidf)  

print(acc_logreg_tfidf)  

cm4 = metrics.confusion_matrix(y_test, pred_logreg_tfidf, labels=[0,1])plot_confusion_matrix(cm4, classes=['TRUE','FAKE'], title ='Confusion matrix for a Logistic Regression with Tf-IDF') 	
		

logreg = LogisticRegression(C=1e5) 

logreg.fit(count_train, y_train) 

pred_logreg_count = logreg.predict(count_test) 

acc_logreg_count = metrics.accuracy_score(y_test,pred_logreg_count) 

print(acc_logreg_count) 

cm3 = metrics.confusion_matrix(y_test, pred_logreg_count, labels=[0,1])plot_confusion_matrix(cm3, classes=['TRUE','FAKE'], title ='Confusion matrix for a Logistic Regression with Count Vectorizer') 

 