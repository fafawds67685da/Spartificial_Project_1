import streamlit as st
import pandas as pd
import requests
import io

# Function to add background image
def add_background_image(image_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url("{image_url}") no-repeat center center fixed;
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Function to fetch predictions
def get_predictions(file):
    files = {'file': file.getvalue()}
    response = requests.post("http://127.0.0.1:8000/predict/", files=files)
    if response.status_code == 200:
        return pd.read_csv(io.StringIO(response.text))
    return None

# Function to fetch the plot
def get_plot(file):
    files = {'file': file.getvalue()}
    response = requests.post("http://127.0.0.1:8000/plot/", files=files)
    if response.status_code == 200:
        return response.content
    return None

# Streamlit app configuration
st.set_page_config(page_title="Dev's First Complete Web Interface 🌟", page_icon="🛐")

# Add background image (Replace URL with your image URL)
background_image_url = "https://astro.cornell.edu/sites/default/files/styles/pano/public/2022-11/CarinaNebulaWebb.jpg?h=42541cb7&itok=ROrFJymI"
add_background_image(background_image_url)

# Main App Title
st.title("Dev's First Complete Web Interface 🌟")

# Add instructions container
st.markdown(
    """
    <div style="background-color: black; padding: 20px; color: white; border-radius: 10px;">
        <h3>Instructions for the Uploaded CSV File:</h3>
        <p>The CSV file should contain two columns:</p>
        <ul>
            <li><strong>Brightness</strong> (First Column) - The brightness of the stars.</li>
            <li><strong>Star Size</strong> (Second Column) - The respective size of the stars.</li>
        </ul>
        <p>Please ensure that the file is formatted correctly before uploading.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# File Upload
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:
    # Predict endpoint
    with st.spinner("Generating predictions..."):
        predicted_df = get_predictions(uploaded_file)
        if predicted_df is not None:
            col1, col2 = st.columns(2)
            with col1:
                st.write("### Original CSV")
                uploaded_df = pd.read_csv(uploaded_file)
                st.dataframe(uploaded_df)
            with col2:
                st.write("### Predicted CSV")
                st.dataframe(predicted_df)
        else:
            st.error("Failed to generate predictions. Please try again.")

    # Plot button and display
    if st.button("Plot the Linear Regression"):
        with st.spinner("Generating plot..."):
            # Send the predicted CSV back for plotting
            plot_content = get_plot(io.BytesIO(predicted_df.to_csv(index=False).encode()))
            if plot_content:
                st.image(plot_content, caption="Linear Regression Plot", use_container_width=True)
            else:
                st.error("Failed to generate the plot. Please try again.")
