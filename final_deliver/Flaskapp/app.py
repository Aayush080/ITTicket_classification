#Import Libraries
from flask import Flask, request, render_template
import tensorflow as tf
import json
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences


app = Flask(__name__)

model = tf.keras.models.load_model("fastTextmodel.h5")
with open('tokenizerfasttext.json') as f:
    data = json.load(f)
    tokenizer = tokenizer_from_json(data)

def predict_ticket(text):
  Description=text
  sequences = tokenizer.texts_to_sequences([Description])
  padded = pad_sequences(sequences,maxlen=120, truncating='post',padding='post')
  predicted=model.predict_classes(padded)
  dict={'0':'Application',"1": 'Database',"2": 'Network', "3":'Security', "4":'User Maintenance'}
  predicted=dict[str(predicted[0])]
  return predicted


# render htmp page
@app.route('/')
def home():
    return render_template('index.html')

# get user input and the predict the output and return to user
@app.route('/predict',methods=['POST'])
def predict():
    
    #take data from form and store in each feature    
    description = str(request.form.values())
    
    
    # predict the price of house by calling model.py
    predicted_ticket_type = predict_ticket(description)     


    # render the html page and show the output
    return render_template('index.html', prediction_text='Predicted Ticket type  is {}'.format(predicted_ticket_type))




if __name__ == "__main__":
    app.run()
