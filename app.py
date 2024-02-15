from flask import Flask, render_template, request
from joblib import load
import pandas as pd

app = Flask(__name__)

# Load the pre-trained models
clf = load('C:/Final Project/Final Project/model_files/bank_deposit_classification.joblib')
ct = load('C:/Final Project/Final Project/model_files/column_transformer.joblib')
label_encoder = load('C:/Final Project/Final Project/model_files/label_encoder.joblib')

def process_data(data, col_names):
    new_data = pd.DataFrame([data], columns=col_names)
    X_new_encoded = pd.DataFrame(ct.transform(new_data))
    y_new_pred = clf.predict(X_new_encoded)
    prediction_label = label_encoder.inverse_transform(y_new_pred)

    return prediction_label

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST', 'GET'])
def submit():
    list_of_col_names = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing',
                         'loan', 'contact', 'day', 'month', 'campaign', 'pdays', 'previous',
                         'poutcome']
    num_cols = ['age', 'balance', 'day', 'campaign', 'pdays', 'previous']
    
    if request.method == 'POST':
        data = [int(request.form[i]) if i in num_cols else request.form[i] for i in list_of_col_names]
        cls_result = process_data(data, list_of_col_names)
        
        # Add a default response in case the method is not POST
        return render_template('index.html', result=cls_result[0])

if __name__ == '__main__':
    app.run(debug=True)
