from flask import Flask, render_template, request, jsonify, send_file, url_for, session, send_from_directory
import pandas as pd
import plotly.express as px
# from visualizations import create_doctor_figure, create_specialty_figure, create_organization_figure, create_doctor_date_figure, create_org_date_figure, create_spec_date_figure
import pickle
from claims import * 
import plotly.graph_objects as go
import interpretability

import matplotlib.pyplot as plt
import os
from flask_session import Session




app = Flask(__name__)

classes= ["Technical", "Doesn't follow clinical practice guidelines"]
app.secret_key = os.urandom(24)  # Required for session to work
app.config['SESSION_TYPE'] = 'filesystem'  # Store session in filesystem
Session(app)  # Initialize session

# Read data

# X_train = pd.read_parquet('15OCT_train.parquet').reset_index().drop(columns = ['index'])
# X_test = pd.read_parquet('filtered_15OCT_test.parquet').reset_index().drop(columns = ['index'])
# y_train = pd.read_parquet('oct_y_train.parquet').reset_index().drop(columns = ['index'])
# y_test = pd.read_parquet('15OCT_y_test.parquet').reset_index().drop(columns = ['index'])
df = pd.read_csv('amr_claims_second_model.csv')#.reset_index(drop= True)
# df = pd.read_csv('claims_filtered_data.csv')


# Load model 
with open("claim_model.pkl", 'rb') as f:
        model = pickle.load(f)
        
# Load interpreter
with open('interpreter.pkl', 'rb') as f:
        interpreter= pickle.load(f)
        
# Path where images are saved
IMAGE_FOLDER = os.path.join('static', 'images')
app.config['IMAGE_FOLDER'] = IMAGE_FOLDER

# Initial Plotly visualizations
# fig_doctors = create_doctor_figure(df_walkin_and_noShow_preds)


# Route for initial page rendering
@app.route('/')
def index():
    
    
    # Assuming df and model are already defined

    # Sample 10 rows from the DataFrame
    inf_df = df.sample(10)

    # Placeholder to store prediction probabilities
    predict_probs = []

    # Loop through rows and predict probabilities
    for index, row in inf_df.iterrows():
        print(index)
        # Reshape the row to a DataFrame as required by your model
        row = row.to_frame().T
        # Append the prediction probabilities (assuming it returns a list or array of probabilities)
        predict_probs.append(model.predict_prob(row, index)[0])  # Assuming predict_prob returns a list of probabilities

    # Convert the predict_probs list into a DataFrame with 2 columns (assuming binary classification)
    predict_probs_df = pd.DataFrame(predict_probs, columns=['Prob_Class_0', 'Prob_Class_1'])

    # Reset indices to ensure they align for concatenation
    inf_df.reset_index(drop=True, inplace=True)
    predict_probs_df.reset_index(drop=True, inplace=True)

    # Concatenate the predictions with the original sampled DataFrame
    inf_df = pd.concat([inf_df, predict_probs_df], axis=1).iloc[:,:10]
    
    session['inf_df'] = inf_df.to_json()  # Convert DataFrame to JSON and store in session
    
    df_html = inf_df.to_html(classes='table table-striped', index=False)

    # Display the updated DataFrame with the prediction probabilities as two new columns
    # print(inf_df)

    
    return render_template('index.html', df_html=df_html)
    
    


@app.route('/generate_image', methods=['POST'])
def generate_image():
    # Get the index from the AJAX request
    row_index = request.json.get('index')

    # Retrieve the DataFrame from the session
    inf_df = pd.read_json(session.get('inf_df'))

    # Access the specific row by index
    # row = inf_df.iloc[int(row_index)]
    row = inf_df[inf_df["Unnamed: 0"] == int(row_index)]
    print(f"Accessing row: {row}")

    # Generate the plot (using matplotlib or your existing method)
    figs = interpreter.plot_contribution(idx=row.index[0], agg=False, max_display=15)
    plt.subplots_adjust(left=0.15, right=0.85, top=0.9, bottom=0.1)

    # Save the figure
    image_file = f"static/plot_{row.index[0]}.png"  # Save the plot as a PNG image
    figs[0].savefig(image_file, dpi=300, bbox_inches='tight')

    # Return the path to the image for the frontend to display
    return jsonify({'image_path': image_file})

@app.route('/static/<filename>')
def send_image(filename):
    return send_from_directory('static', filename)








if __name__ == '__main__':
    app.run(debug=True)
