from traceback import print_tb
from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__, template_folder='templates')

with open('logistic_model.pkl', 'rb') as file:  
    pickle_file = pickle.load(file)


spicies_encodings = {
    
    0: 'Bream',
    1: 'Parkki',
    2: 'Perch',
    3: 'Pike',
    4: 'Roach',
    5: 'Smelt',
    6: 'Whitefish',
}


@app.route('/', methods=['GET', 'POST'])
def index():
  if request.method == 'POST':
    form = request.form
    value = classify(form)
    return render_template('home.html', result=spicies_encodings[value[0]])
  
  else:
    return render_template('home.html')




def classify(form):
  df = pd.DataFrame(data = [{
            
            'Weight':  float(form['Weight']), 
            'Length1': float(form['Length1']), 
            'Length2': float(form['Length2']), 
            'Length3': float(form['Length3']), 
            'Height':  float(form['Height']), 
            'Width':   float(form['Width'])
      }])
    
  return pickle_file.predict(df)

if __name__ == "__main__":
  app.run(debug=True)
