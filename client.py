# client.py
from flask import Flask, render_template, jsonify, request, session
import requests
from flask_session import Session

import os

import pandas as pd


app = Flask(__name__)
app.secret_key = os.urandom(24)  # Required for session to work
app.config['SESSION_TYPE'] = 'filesystem'  # Store session in filesystem
Session(app)  # Initialize session

# URL of the server API
SERVER_URL = 'http://127.0.0.1:5000'  # Update if server is hosted elsewhere

@app.route('/')
def index():
    response = requests.post(f'{SERVER_URL}/send_filter_values')
    specs , dates, PUR_NAMEs, POLICY_NAMEs = response.json()
    session["filters_all"] = response.json()
    return render_template('index3.html', specs=specs, dates=dates, PUR_NAMEs=PUR_NAMEs, POLICY_NAMEs=POLICY_NAMEs)

    

@app.route('/get_filter_values', methods=['POST'])
def get_filter_values():
    # Forward request to server API
    data = request.get_json()
    session["selected_filters"] = data
    response = requests.post(f'{SERVER_URL}/get_filter_values', json=data)
    return jsonify(response.json())



@app.route('/apply_filters', methods=['POST'])
def apply_filters():
    # Forward request to server API to apply filters
    data = request.get_json()
    print(data)
    response = requests.post(f'{SERVER_URL}/apply_filters', json=data)
    print(response.status_code, "***&&^%$####")
    if response.status_code == 200:
        # Return the HTML table for display
        # response
        df_html = pd.DataFrame(response.json()).to_html(classes='table table-striped', index=False)
        session["filterd_df"] = pd.DataFrame(response.json()).to_json()
        return jsonify({'df_html': df_html})
    else:
        return jsonify({"error": "Failed to apply filters"}), 500

@app.route('/generate_image', methods=['POST'])
def generate_image():
    # Forward request to server API to generate an image for the selected row
    # data = request.get_json()
    # print(int(request.json.get('index')))
    # data = pd.read_json(session["filterd_df"])
    # data = [data[data["Unnamed: 0"] == int(request.json.get('index'))].to_json() , request.json.get('index')] 
    data = [session["filterd_df"], request.json.get('index')]
    response = requests.post(f'{SERVER_URL}/generate_image', json=data)
    if response.status_code == 200:
        # Return the image path for display in the client
        return jsonify(response.json())
    else:
        return jsonify({"error": "Failed to generate image"}), 500

if __name__ == '__main__':
    app.run(port=5005, debug=True)  # Run the client app on port 5001
