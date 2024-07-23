import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Load the trained model from the pickle file
model_filename = 'C:/Users/thoms/Desktop/TRAVEL/model (1).pkl'  # Adjust path for deployment
with open(model_filename, 'rb') as file:
    model = pickle.load(file)

# Load the dataset to get feature names and destinations
data_filename = 'C:/Users/thoms/Desktop/TRAVEL/synthetic_travel_data_with_starting_destination.csv'  # Adjust path for deployment
df = pd.read_csv(data_filename)

# Prepare columns for user input
all_features = df.columns[df.columns != 'Preferred Destination']  # Exclude the target column
unique_destinations = df['Preferred Destination'].unique()

# Streamlit app interface
st.title("Intelligent Travel Planning System")
st.write("Enter your income, starting destination, and preferred destination to get predictions for travel plans:")

# Input fields for user income, starting destination, and preferred destination
user_income = st.number_input("User Income (USD)", min_value=0, step=100)
starting_destination = st.selectbox("Starting Destination", options=df['Starting Place'].unique())
preferred_destination = st.selectbox("Preferred Destination", options=unique_destinations)

# Predict button
if st.button("Get Travel Plans"):
    # Create an input dictionary with the user income, starting destination, and preferred destination
    input_data = {
        'User Income': user_income,
        'Starting Destination': starting_destination,
        'Preferred Destination': preferred_destination,
    }

    # Ensure that the input data contains all required features
    for feature in all_features:
        if feature not in input_data:
            input_data[feature] = 0  # Placeholder value for missing features

    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])

    # Preprocessing
    # Handle categorical variables
    input_df_encoded = pd.get_dummies(input_df)
    
    # Ensure the order of columns matches the model's expected features
    input_df_encoded = input_df_encoded.reindex(columns=model.feature_names_in_, fill_value=0)
    
    # Impute missing values (adjust if using a specific imputer)
    imputer = SimpleImputer(strategy='mean')  # Ensure imputer is consistent with training
    input_df_encoded = pd.DataFrame(imputer.fit_transform(input_df_encoded), columns=input_df_encoded.columns)
    
    # Standardize the input data (adjust if using a specific scaler)
    scaler = StandardScaler()  # Ensure scaler is consistent with training
    input_df_encoded = pd.DataFrame(scaler.fit_transform(input_df_encoded), columns=input_df_encoded.columns)

    # Make predictions
    try:
        # Generate predictions
        predictions = model.predict(input_df_encoded)
        
        # Filter the dataset based on the user income and preferred destination
        filtered_df = df[
            (df['User Income'].between(user_income - 10000, user_income + 10000)) &
            (df['Preferred Destination'] == preferred_destination) &
            (df['Starting Place'] == starting_destination)
        ]

        # Display results in a card format, limiting to 10 results
        if not filtered_df.empty:
            st.write("Travel Plans within your income range:")
            
            # Display results in card format
            cols = st.columns(2)  # Create two columns for card layout
            for index, row in filtered_df.head(10).iterrows():
                with cols[index % 2]:  # Use modulo to alternate columns
                    st.markdown(
                        f"""
                        <div style="border: 1px solid #ddd; border-radius: 5px; padding: 10px; margin: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
                            <h4>Destination: {row['Preferred Destination']}</h4>
                            <p><strong>Accommodation Type:</strong> {row['Accommodation Type']}</p>
                            <p><strong>Travel Mode:</strong> {row['Preferred Travel Mode']}</p>
                            <p><strong>Travel Duration:</strong> {row['Travel Duration (days)']} days</p>
                            <p><strong>Accommodation Budget:</strong> ${row['Accommodation Budget (USD)']}</p>
                            <p><strong>Activity Budget:</strong> ${row['Activity Budget (USD)']}</p>
                            <p><strong>Number of Travelers:</strong> {row['Number of Travelers']}</p>
                            <p><strong>Total Budget:</strong> ${row['Total Budget (USD)']}</p>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
        else:
            st.write("No travel plans available within this income range.")
    
    except Exception as e:
        st.write(f"Error during prediction: {e}")
