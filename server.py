from flask import Flask, jsonify, request, session, send_from_directory, send_file
import base64
import pandas as pd
import pickle
from flask_session import Session
import os
import matplotlib.pyplot as plt
from io import BytesIO
from datetime import timedelta
import uuid
pd.options.mode.chained_assignment = None  # Turn off SettingWithCopyWarning

import warnings

# Suppress FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['SESSION_TYPE'] = 'filesystem'
app.secret_key = os.urandom(24)
app.config['SESSION_TYPE'] = 'filesystem'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=24)
Session(app)

# Load data and model
df = pd.read_csv('amr_claims_second_model.csv')
with open("claim_model.pkl", 'rb') as f:
    model = pickle.load(f)
with open('interpreter.pkl', 'rb') as f:
    interpreter = pickle.load(f)
    
    
    # User session management
def init_user_session():
    """Initialize user session if not already present"""
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
    if 'filtered_df' not in session:
        session['filtered_df'] = None
    session.permanent = True

# Define the filter function
def filter_dataframe(dataframe, filters):
    filtered_df = dataframe.copy()
    if filters.get('spec'):
        filtered_df = filtered_df[filtered_df['DOCTOR_SPECIALTY_CODE'] == float(filters['spec'])]
        
        
    # if filters.get('date'):
    #     filtered_df = filtered_df[filtered_df['CREATION_DATE'] == filters['date']]
    if filters.get("start_date") or filters.get("start_date"):
        print("date")
        # Retrieve start_date and end_date from the selected_values dict
        start_date = filters['start_date']
        end_date = filters['end_date']

        # print(start_date, end_date)  # Debug: log the date values

        # If both dates are provided, filter the DataFrame by date range
        if start_date and end_date:
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date) + pd.Timedelta(days=1) 

            # Filter DataFrame by date range
            filtered_df['CREATION_DATE'] = pd.to_datetime(filtered_df['CREATION_DATE'])
            filtered_df = filtered_df[(filtered_df['CREATION_DATE'] >= start_date) & (filtered_df['CREATION_DATE'] <= end_date)]
        
    if filters.get('PUR_NAME'):
        filtered_df = filtered_df[filtered_df['PUR_NAME'] == filters['PUR_NAME']]
        print("HEREEEE&*(&*(&**(&*())))")
        print(filtered_df["PUR_NAME"].value_counts())
        
    
    if filters.get('POLICY_NAME'):
        filtered_df = filtered_df[filtered_df['POLICY_NAME'] == filters['POLICY_NAME']]
    return filtered_df


@app.route('/get_filter_values', methods=['POST'])
def get_filter_values():
    
    init_user_session()
    data = request.json
    # print(data)  # Debug: log the incoming request data
    

    filter_column =  data['filter_column']
    selected_values = data['selected_values']

    # Default to return an empty list if no filter_column is found
    unique_values = []

    # Check the filter column type
    if filter_column == "spec":
        filter_column = "DOCTOR_SPECIALTY_CODE"
    elif filter_column == "start_date" or filter_column == "end_date":
        filter_column = "CREATION_DATE"
        
        # Retrieve start_date and end_date from the selected_values dict
        start_date = data["selected_values"]['start_date']
        end_date = data["selected_values"]['end_date']

        # print(start_date, end_date)  # Debug: log the date values

        # If both dates are provided, filter the DataFrame by date range
        if start_date and end_date:
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)

            # Filter DataFrame by date range
            df['CREATION_DATE'] = pd.to_datetime(df['CREATION_DATE'])
            filtered_df = df[(df['CREATION_DATE'] >= start_date) & (df['CREATION_DATE'] <= end_date)]
            
            # If there are any records in the filtered DataFrame, extract unique values
            if not filtered_df.empty:
                unique_values = filtered_df[filter_column].unique().tolist()
    
    else:
        # For other filters, filter the DataFrame by selected values
        filtered_df = filter_dataframe(df, selected_values)
        
        
        # If the filtered DataFrame has rows, extract unique values
        # if not filtered_df.empty:
        #     unique_values = filtered_df[filter_column].unique().tolist()

    # Return the unique values as a JSON response (even if it's an empty list)
    return jsonify(unique_values)


@app.route('/send_filter_values', methods=['POST'])
def send_filter_values():
    init_user_session()
    specs = df['DOCTOR_SPECIALTY_CODE'].unique().tolist()
    dates = df['CREATION_DATE'].unique().tolist()
    
    PUR_NAMEs = df['PUR_NAME'].unique().tolist()
    POLICY_NAMEs = df['POLICY_NAME'].unique().tolist()
    
    
    return jsonify([specs, dates, PUR_NAMEs, POLICY_NAMEs])





@app.route('/apply_filters', methods=['POST'])
def apply_filters():
    init_user_session()
    selected_values = request.json
    print(selected_values, "999999")
    filtered_df = filter_dataframe(df, selected_values)
    filtered_df.to_csv("filtered_df.csv")
    
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
        # print(row_index_og.index[0], "AFTER PRED ****************")
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
    # df_html = selected_columns.to_html(classes='table table-striped', index=False)
    
    return jsonify(selected_columns.to_dict())



@app.route('/generate_image', methods=['POST'])
def generate_image():
    init_user_session()
    row_index = request.json[1]
    # print(session["_permanent"])
    inf_df = pd.read_json(request.json[0])
    # print(inf_df)
    # print(request.json)
    row = inf_df[inf_df["Unnamed: 0"] == int(row_index)]
    figs = interpreter.plot_contribution(idx=row.index[0], P=inf_df.iloc[row.index[0]]["Prob_Class_1"], agg=False, max_display=15)
    plt.subplots_adjust(left=0.15, right=0.85, top=0.9, bottom=0.1)

    # Save the figure
    # image_file = f"static/plot_{row.index[0]}_here.png"
    # figs[0][0].savefig(image_file, dpi=300, bbox_inches='tight')
    
    # Save plot to an in-memory buffer
    buffer = BytesIO()
    figs[0][0].savefig(buffer, format='png', dpi=300, bbox_inches='tight')
    buffer.seek(0)
    
    # Encode the image as base64
    image_data = base64.b64encode(buffer.read()).decode('utf-8')
    return jsonify({'image_data': image_data})  # Return base64 image data
    # return send_file(buffer, mimetype='image/png', as_attachment=False)

    # return jsonify({'image_path': f"{image_file}"})

@app.route('/static/<filename>')
def send_image(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")
