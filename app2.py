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
import warnings
# Load your data and model
df = pd.read_csv('amr_claims_second_model.csv')
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)

app = Flask(__name__)

# ... (keep your existing imports and configurations)
classes= ["Technical", "Doesn't follow clinical practice guidelines"]
app.secret_key = os.urandom(24)  # Required for session to work
app.config['SESSION_TYPE'] = 'filesystem'  # Store session in filesystem
Session(app)  # Initialize session



with open("claim_model.pkl", 'rb') as f:
    model = pickle.load(f)
    
# Load interpreter
with open('interpreter.pkl', 'rb') as f:
        interpreter= pickle.load(f)
        
        
# DOCTOR_SPECIALTY_CODE, CREATION_DATE, PUR_NAME, POLICY_NAME
def filter_dataframe(dataframe, filters):
    filtered_df = dataframe.copy()
    if filters.get('spec'):
        filtered_df = filtered_df[filtered_df['DOCTOR_SPECIALTY_CODE'] == filters['spec']]
    if filters.get('date'):
        filtered_df = filtered_df[filtered_df['CREATION_DATE'] == filters['date']]
    if filters.get('PUR_NAME'):
        filtered_df = filtered_df[filtered_df['PUR_NAME'] == filters['PUR_NAME']]
    if filters.get('POLICY_NAME'):
        filtered_df = filtered_df[filtered_df['POLICY_NAME'] == filters['POLICY_NAME']]
                
    return filtered_df

@app.route('/')
def index():
    # Get initial filter values
    specs = df['DOCTOR_SPECIALTY_CODE'].unique().tolist()
    dates = df['CREATION_DATE'].unique().tolist()
    PUR_NAMEs = df['PUR_NAME'].unique().tolist()
    POLICY_NAMEs = df['POLICY_NAME'].unique().tolist()
    return render_template('index2.html', specs=specs, dates=dates, PUR_NAMEs=PUR_NAMEs, POLICY_NAMEs=POLICY_NAMEs)

@app.route('/get_filter_values', methods=['POST'])
def get_filter_values():
    filter_column = request.json['filter_column']
    selected_values = request.json['selected_values']

    filtered_df = filter_dataframe(df, selected_values)
    unique_values = filtered_df[filter_column].unique().tolist()
    return jsonify(unique_values)

@app.route('/apply_filters', methods=['POST'])
def apply_filters():
    selected_values = request.json

    filtered_df = filter_dataframe(df, selected_values)
    
    # Sample 10 rows from the filtered DataFrame
    sampled_df = filtered_df.sample(min(10, len(filtered_df)))
    
    # Predict probabilities
    predict_probs = []
    for _, row in sampled_df.iterrows():
        row_df = row.to_frame().T
        predict_probs.append(model.predict_prob(row_df, row.name)[0])
    
    # Add prediction probabilities to the dataframe
    predict_probs_df = pd.DataFrame(predict_probs, columns=['Prob_Class_0', 'Prob_Class_1'])
    result_df = pd.concat([sampled_df.reset_index(drop=True), predict_probs_df], axis=1)
    
    # Store the filtered DataFrame in the session
    session['filtered_df'] = result_df.to_json()
    
    selected_columns = result_df.iloc[:, :10].copy()  # First 10 columns
    selected_columns[['Prob_Class_0', 'Prob_Class_1']] = result_df[['Prob_Class_0', 'Prob_Class_1']]  # Add the specific columns
    
    
    # Loop through the DataFrame
    for index, row in selected_columns.iterrows():
        # print()
        # print(f"Index: {index}")
        # print(f"Row data:\n{row}\n")
        
        row_index_og = result_df[result_df["Unnamed: 0"] == int(row["Unnamed: 0"])]
        print(row_index_og.index[0], "AFTER PRED ****************")
        figs = interpreter.plot_contribution(idx=row_index_og.index[0], P=row["Prob_Class_1"], agg=False, max_display=15, plot=False)
        new_data = figs[1][0][['SERVICE_DESCRIPTION_Agg', 'DIAGNOSIS_Agg', 'SIGNS_AND_SYMPTOMS_Agg', 'CPG_COMPLIANCE', 
                 'INSURANCE_COMPANY', 'SUB_ACCOUNT', 'CONTRACT_NO (MEMBER_CLASS)', 'TREATMENT_TYPE', 'PROVIDER_DEPARTMENT_CODE',
                 'CLAIM_TYPE']].iloc[0]
        
        # print(figs[1][0].to_csv("a.csv"))
        
         # Update the current row with new values
        for col in new_data.index:
            # print(col, index)
            selected_columns.loc[index, col] = new_data[col]
    
    # session['selected_columns'] = selected_columns.to_json()
    
    # Return only the first 10 rows without predictions
    df_html = selected_columns.to_html(classes='table table-striped', index=False)
    return jsonify({'df_html': df_html})




@app.route('/run_prediction', methods=['POST'])
def run_prediction():
    # Retrieve the filtered DataFrame from the session
    filtered_df = pd.read_json(session.get('filtered_df'))
    
    # Sample 10 rows from the filtered DataFrame
    sampled_df = filtered_df.sample(min(10, len(filtered_df)))
    
    # Predict probabilities
    predict_probs = []
    for _, row in sampled_df.iterrows():
        row_df = row.to_frame().T
        predict_probs.append(model.predict_prob(row_df, row.name)[0])
    
    # Add prediction probabilities to the dataframe
    predict_probs_df = pd.DataFrame(predict_probs, columns=['Prob_Class_0', 'Prob_Class_1'])
    result_df = pd.concat([sampled_df.reset_index(drop=True), predict_probs_df], axis=1)
    
   
    


     # Store the result in the session
    session['inf_df'] = result_df.to_json()
    
    
    # Return the HTML table with predictions
    df_html = result_df.to_html(classes='table table-striped', index=False)
    return jsonify({'df_html': df_html})

# ... (keep your existing generate_image and send_image routes)


@app.route('/generate_image', methods=['POST'])
def generate_image():
    # Get the index from the AJAX request
    row_index = request.json.get('index')

    # Retrieve the DataFrame from the session
    inf_df = pd.read_json(session.get('filtered_df'))

    # Access the specific row by index
    # row = inf_df.iloc[int(row_index)]
    row = inf_df[inf_df["Unnamed: 0"] == int(row_index)]
    # print(f"Accessing row: {row}")
    
    # print(inf_df)

    
    # Generate the plot (using matplotlib or your existing method)
    # figs = interpreter.plot_contribution(idx=row.index[0], agg=False, max_display=15)
    print(row.index[0], "PLOTTING $$$$$$$$$$$$$$$$$$$$")
    figs = interpreter.plot_contribution(idx=row.index[0], P=inf_df.iloc[row.index[0]]["Prob_Class_1"], agg=False, max_display=15)
    plt.subplots_adjust(left=0.15, right=0.85, top=0.9, bottom=0.1)

    # Save the figure
    image_file = f"static/plot_{row.index[0]}.png"  # Save the plot as a PNG image
    figs[0][0].savefig(image_file, dpi=300, bbox_inches='tight')

    # Return the path to the image for the frontend to display
    return jsonify({'image_path': image_file})

@app.route('/static/<filename>')
def send_image(filename):
    return send_from_directory('static', filename)



if __name__ == '__main__':
    app.run(debug=True)