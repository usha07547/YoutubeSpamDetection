from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

app = Flask(__name__, template_folder='template')

# Load and preprocess data once when the server starts
df = pd.read_csv("D:/FlaskApp/WebApp/Youtube-Spam-Dataset.csv")
df_data = df[["CONTENT", "CLASS"]]
df_x = df_data['CONTENT']
df_y = df_data['CLASS'].values

# Vectorization and model training
cv = CountVectorizer()
X = cv.fit_transform(df_x)
X_train, X_test, y_train, y_test = train_test_split(X, df_y, test_size=0.33, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)
# You can print accuracy if needed: print(model.score(X_test, y_test))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        comment = request.form['comment']
        data = [comment]
        vect = cv.transform(data).toarray()
        my_prediction = model.predict(vect)
        return render_template('results.html', prediction=my_prediction[0])
if __name__ == '__main__':
    app.run(debug=False)
