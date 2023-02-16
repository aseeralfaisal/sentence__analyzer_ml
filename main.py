from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

categories = ["alt.atheism", "soc.religion.christian", "comp.graphics", "sci.med"]

twenty_train = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)
twenty_train.target_names
["alt.atheism", "comp.graphics", "sci.med", "soc.religion.christian"]

countVectorizer = CountVectorizer()
X_train_counts = countVectorizer.fit_transform(twenty_train.data)
tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)

val = input('Type a sentence \n')
input_data = []
input_data.append(val)
X_new_counts = countVectorizer.transform(input_data)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

predicted = clf.predict(X_new_tfidf)

for doc, category in zip(input_data, predicted):
    prediction = twenty_train.target_names[category] 
    if prediction == "alt.atheism" or prediction == "soc.religion.christian":
        result = "Religion Related"
    elif prediction == "sci.med":
        result = "Medical Related"
    else:
        result = "Graphics Related"
    print(prediction)
    print("%r => %s" % (doc, result))

