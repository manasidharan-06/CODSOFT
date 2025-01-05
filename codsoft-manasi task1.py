import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
import joblib

# Loading the datasets
x=train_data = pd.read_csv("train_data.txt", sep=":::", names=["title", "genre", "des"], engine="python")
y=test_data = pd.read_csv("test_data.txt", sep=":::", names=["id", "title", "des"], engine="python")

# Dropping missing values
x.dropna(subset=["des", "genre"], inplace=True)
y.dropna(subset=["des"], inplace=True)

# Transforming training data and testing data using TF_IDF
TF_IDF = TfidfVectorizer()
x_train = TF_IDF.fit_transform(x["des"])  # Transform training data
x_test = TF_IDF.transform(y["des"])       # Transform test data
y_train = x["genre"]

# Training the model
model = MultinomialNB()
model.fit(x_train, y_train)

# Saving the model and TF-IDF vectorizer
joblib.dump(model, "movie_genre_model.pkl")
joblib.dump(TF_IDF, "tfidf_vectorizer.pkl")

# Predicting on the train data for evaluation
y_pred = model.predict(x_train)
accuracy = accuracy_score(y_train, y_pred)
print(f"\nModel Accuracy on Training Data: {accuracy:.2f}\n")

# Predicting genres for the test data
test_data["predicted_genre"] = model.predict(x_test)

# Saving predictions to CSV
test_data.to_csv("predicted-genre.csv", index=False)
print("Predicted genres saved to 'predicted-genre.csv'.")


