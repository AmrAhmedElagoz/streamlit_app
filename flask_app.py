from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import plotly.express as px
# from visualizations import create_doctor_figure, create_specialty_figure, create_organization_figure, create_doctor_date_figure, create_org_date_figure, create_spec_date_figure
import pickle
from claims import * 
import plotly.graph_objects as go
import interpretability

import matplotlib.pyplot as plt




app = Flask(__name__)

classes= ["Technical", "Doesn't follow clinical practice guidelines"]


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
        
        

# Initial Plotly visualizations
# fig_doctors = create_doctor_figure(df_walkin_and_noShow_preds)


# Route for initial page rendering
@app.route('/')
def index():
    
    inf_df = df.sample()
    predict_probs = model.predict_prob(inf_df, inf_df.index[0])
    fig = go.Figure(data=[go.Pie(values=predict_probs[0], labels=classes)])
    figs = interpreter.plot_contribution(idx=inf_df.index[0], agg=False, max_display=15)
    plt.subplots_adjust(left=0.15, right=0.85, top=0.9, bottom=0.1)

    # Save the figure
    file = "plot99.png"
    figs[0].savefig(file, dpi=300, bbox_inches='tight')
    
    fig.write_image("figure.png")
    
    return render_template('index.html',
                           fig_doctors=fig.to_html(),
                           img1 = "figure.png",
                           plot_img=file
                           )
    
@app.route('/images/<filename>')
def send_image(filename):
    # Serve the image from the given file path
    return send_file(filename)

# # Route for fetching filter choices
# @app.route('/get_filter_choices', methods=['GET'])
# def get_filter_choices():
#     # # Assuming 'doctor_name', 'unified_specialty', 'organization_key' for filters
#     # org_key_choices = list(df_walkin_and_noShow_preds['organization_key'].unique())
#     # uni_specialty_choices = list(df_walkin_and_noShow_preds['unified_specialty'].unique())
#     # doctor_name_choices = list(df_walkin_and_noShow_preds['doctor_name'].unique())

#     # return jsonify(org_key=org_key_choices, uni_specialty=uni_specialty_choices, doctor_name=doctor_name_choices)

# # Route for updating plots via AJAX
# @app.route('/update_plots', methods=['POST'])
# def update_plots():
#     org_key_value = request.form.get('org_key')
#     uni_specialty_value = request.form.get('uni_specialty')
#     doctor_name_value = request.form.get('doctor_name')

#     # Update data based on filters (replace with your own logic)
#     updated_data = update_data_based_on_filters(org_key_value, uni_specialty_value, doctor_name_value)
#     doctor_name_choices = list(updated_data['doctor_name'].unique())

#     # Update Plotly visualizations
#     fig_doctors = create_doctor_figure(updated_data)
#     fig_specialties = create_specialty_figure(updated_data)
#     fig_organizations = create_organization_figure(updated_data)
#     fig_doctors_date = create_doctor_date_figure(updated_data)
#     fig_org_date = create_org_date_figure(updated_data)
#     fig_spec_date = create_spec_date_figure(updated_data)

#     # Return HTML representations of updated plots
#     return jsonify(fig_doctors=fig_doctors.to_html(),
#                    fig_specialties=fig_specialties.to_html(),
#                    fig_organizations=fig_organizations.to_html(),
#                    fig_doctors_date=fig_doctors_date.to_html(),
#                    fig_org_date=fig_org_date.to_html(),
#                    fig_spec_date=fig_spec_date.to_html(),
#                    doctor_name=doctor_name_choices)

# # Replace with your actual data manipulation logic
# def update_data_based_on_filters(org_key, uni_specialty, doctor_name):
#     updated_data = df_walkin_and_noShow_preds.copy()

#     if org_key != "All" and org_key:
#         updated_data = updated_data[updated_data['organization_key'] == org_key]

#     if uni_specialty != "All" and uni_specialty:
#         updated_data = updated_data[updated_data['unified_specialty'] == uni_specialty]

#     if doctor_name != "All" and doctor_name:
#         updated_data = updated_data[updated_data['doctor_name'] == doctor_name]

#     return updated_data

if __name__ == '__main__':
    app.run(debug=True)
