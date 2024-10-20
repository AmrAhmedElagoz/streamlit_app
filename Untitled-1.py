

# Load Data

X_train = pd.read_parquet('15OCT_train.parquet').reset_index().drop(columns = ['index'])
X_test = pd.read_parquet('filtered_15OCT_test.parquet').reset_index().drop(columns = ['index'])
y_train = pd.read_parquet('oct_y_train.parquet').reset_index().drop(columns = ['index'])
y_test = pd.read_parquet('15OCT_y_test.parquet').reset_index().drop(columns = ['index'])
df = pd.read_csv('amr_claims_second_model.csv')#.reset_index(drop= True)
# df = pd.read_csv('claims_filtered_data.csv')


# header name
# st.header("Approvals and Rejections Prediction")


# FILTERS 
# st.write("### Filters")
# start_date = st.date_input('Start Date')
# end_date = st.date_input('End Date')


