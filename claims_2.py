import streamlit as st
import pandas as pd
import streamlit as st
import pandas as pd, random
from claims import claim_inference, model
import pickle

from interpretability import shap_lime, Interpretability
import plotly.graph_objects as go

X_train = pd.read_parquet('15OCT_train.parquet').reset_index().drop(columns = ['index'])
X_test = pd.read_parquet('15OCT_test.parquet').reset_index().drop(columns = ['index'])
y_train = pd.read_parquet('oct_y_train.parquet').reset_index().drop(columns = ['index'])
y_test = pd.read_parquet('oct_y_test.parquet').reset_index().drop(columns = ['index'])
df = pd.read_csv('amr_claims.csv')#.reset_index(drop= True)
# df = pd.read_csv('claims_filtered_data.csv')


st.header("Approvals and Rejections Prediction")

# Filter by Start Date and End Date
st.write("### Filters")
start_date = st.date_input('Start Date')
end_date = st.date_input('End Date')

# Apply the date filter if both dates are selected
if start_date and end_date:
    df = df[(pd.to_datetime(df['CREATION_DATE']) >= pd.to_datetime(start_date)) & 
            (pd.to_datetime(df['CREATION_DATE']) <= pd.to_datetime(end_date))]

# Filter by Speciality in the main layout
speciality = st.multiselect("Select Speciality", df['PROVIDER_DEPARTMENT'].unique())
if speciality:
    df = df[df['PROVIDER_DEPARTMENT'].isin(speciality)]

# Filter by Insurance Company in the main layout
insurance = st.multiselect("Select Insurance Company", df['PUR_NAME'].unique())
if insurance:
    df = df[df['PUR_NAME'].isin(insurance)]

# Display the filtered table (without PREDICTED_REASON column)
st.write("## Filtered Data")
f_data= df.drop(columns=['PREDICTED_REASON','ACTUAL_OUTCOME','ACTUAL_REASON'])
st.dataframe(f_data)

# Calculate and display percentage of approved vs rejected outcomes
total_rows = len(df)
approved_count = len(df[df['PREDICTED_OUTCOME'] == 1])
rejected_count = total_rows - approved_count

if total_rows > 0:
    approved_percentage = (approved_count / total_rows) * 100
    rejected_percentage = 100 - approved_percentage
    st.write(f"### Predicted Outcome: {approved_percentage:.2f}% Approved, {rejected_percentage:.2f}% Rejected")
else:
    st.write("No data to display for this filter combination.")

# SHAP
feature_contribution= st.checkbox("Feature Contribution")
# classes= ["Technical", "Doesn't follow clinical practice guidelines"]
classes= ["Rejected", "Approved"]
cfg= {}
cfg['task_type']= "classification"

### To be set
cfg['model']= "approval_rejection.pkl" # rejection_reasons.pkl or approval_rejection.pkl
cfg['interpreter']= "approval_rejection_interpreter.pkl" # rejection_reasons_interpreter.pkl
                                                        # approval_rejection_interpreter.pkl

try:
    with open(cfg['model'], 'rb') as f:
        m = pickle.load(f)

except FileNotFoundError:
    print("Model file not found, retraining...")
    model(X_train, X_test, y_train, y_test, False)
    print("finished training...")
    with open(cfg['model'], 'rb') as f:
        m= pickle.load(f)
        print(f"{cfg['model']} model loaded!")

except pickle.UnpicklingError:
    print("Error loading the pickle model.")

re_shap= st.checkbox("Recalculate SHAP...")

if re_shap:
    print("Calculating SHAP...")
    interpreter = Interpretability(m, cfg['task_type'], X_train, X_test, y_train, y_test, apply_prior= True)
    with open(cfg['interpreter'], 'wb') as f:
        pickle.dump(interpreter, f)
        print("Interpreter model saved!")
else:
    with open(cfg['interpreter'], 'rb') as f:
        interpreter= pickle.load(f)

# X_test= X_test.drop(columns= ['NOTES'])

if feature_contribution:
    if st.checkbox("Generate Random Index"):
        # X_test= f_data
        random_number = st.text_input("inter the number: ")
        random_number= int(random_number)
        _dataframe= X_test.iloc[[random_number]]
        # dis_df= f_data.iloc[[random_number]]
        st.write(_dataframe)
        # st.write(dis_df)

        proba_preds= claim_inference(_dataframe, X_train, X_test, y_train, y_test, random_number, cfg)
        st.write("Model Probability Prediciton")
        st.write(proba_preds)
        if classes:
            fig = go.Figure(data=[go.Pie(values=proba_preds[0], labels=classes)])
        else:
            fig = go.Figure(data=[go.Pie(values=proba_preds[0])])
        st.plotly_chart(fig)

        figs= shap_lime(interpreter, plot_contribution= {'idx': random_number, 'agg': False, 'max_display': 20, 'P': proba_preds[0][0]})
        for fig in figs[0][0]:
            st.pyplot(fig)

        for tab in figs[0][1]:
            tab.index= ["Contribution Percentage %", "Data"]
            tab_dis= tab[['SERVICE_DESCRIPTION_Agg', 'DIAGNOSIS_Agg', 'SIGNS_AND_SYMPTOMS_Agg', 'CPG_COMPLIANCE', 
                 'INSURANCE_COMPANY', 'SUB_ACCOUNT', 'CONTRACT_NO (MEMBER_CLASS)', 'TREATMENT_TYPE', 'PROVIDER_DEPARTMENT_CODE',
                 'CLAIM_TYPE']]
            st.write(tab_dis.T)


# Display the table for rejected outcomes only
rejected_df = df[df['PREDICTED_OUTCOME'] == 'Rejected']

st.write("## Rejected Outcomes with Predicted Reason")
st.dataframe(rejected_df[['CREATION_DATE', 'PUR_NAME', 'POLICY_NAME', 
                          'PROVIDER_DEPARTMENT', 'SERVICE_DESCRIPTION', 'ICD10', 'DIAGNOSIS',
                          'SIGNS_AND_SYMPTOMS', 'CPG_COMPLIANCE', 'PREDICTED_REASON']])

# Calculate and display percentage of medical vs non-medical reasons for rejected outcomes
if not rejected_df.empty:
    medical_count = len(rejected_df[rejected_df['PREDICTED_REASON'] == 'Medical']) 
    non_medical_count = len(rejected_df[rejected_df['PREDICTED_REASON'] == 'Non_Medical'])  
    total_rejected = len(rejected_df)

    medical_percentage = (medical_count / total_rejected) * 100
    non_medical_percentage = 100 - medical_percentage

    st.write(f"### Rejected Reason: {medical_percentage:.2f}% Medical, {non_medical_percentage:.2f}% Non-medical")
else:
    st.write("No rejected outcomes to display.")


