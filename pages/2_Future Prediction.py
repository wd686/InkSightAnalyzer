
import streamlit as st
import numpy as np
import pandas as pd
from controller import controllerService

# Set page config
st.set_page_config(
    page_title='Future Prediction', # Set display name of browser tab
    layout="wide", # "wide" or "centered"
    initial_sidebar_state="expanded"
)

st.header('Aspect Based Sentiment Analysis (ABSA) System')

st.write(f"*Simply download the template CSV file, change the Time Period and Reviews of interest, and upload the modified CSV file.*")
st.write(f"*ABSA models will run in the background and results will be displayed. CSV results can be downloaded.*")

data = {
    "Time period": ["April - May 2024"] + [""] * 8,
    "Reviews": [
        "Review sample 1", "Review sample 2", "Review sample 3",
        "Review sample 4", "Review sample 5", "Review sample 6",
        "Review sample 7", "Review sample 8", "Review sample 9"
    ]
}

# Create the DataFrame
template_df = pd.DataFrame(data)

st.subheader('Download your template CSV file here:')

st.download_button("Download template CSV file",
                    template_df.to_csv(index = False),
                    file_name = 'template_file.csv',
                    mime = 'text/csv')

st.subheader('Upload your modified CSV file here:')

uploaded_file = st.file_uploader("Upload CSV File (there should only be 1 cell input for column 'Time period')")

try:
    rawInput_df = pd.read_csv(uploaded_file)

    modelResults = controllerService()
    aspect_df = modelResults.runAspectClassification(rawInput_df)
    aspectSentimentOutput_df, overallResultsOutput_df = modelResults.runSentimentExtraction(aspect_df)

    col1, col2 = st.columns([1,1])

    with col1:
        st.write("Placeholder1 (Wordcloud Visualization)") # TODO insert word-cloud
    with col2:
         st.write("Placeholder2 (Treemap Visualization)") # TODO insert tree map

    st.download_button("Download Aspect-Sentiment Output CSV file",
                    aspectSentimentOutput_df.to_csv(index = False),
                    file_name = 'aspectSentimentOutput_file.csv',
                    mime = 'text/csv')
    
    st.download_button("Download Overall Results Output CSV file",
                    overallResultsOutput_df.to_csv(index = False),
                    file_name = 'overallResultsOutput_file.csv',
                    mime = 'text/csv')

except ValueError:
    pass