
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import squarify
import matplotlib
from controller import controllerService

# Set page config
st.set_page_config(
    page_title='Future Prediction', # Set display name of browser tab
    layout="wide", # "wide" or "centered"
    initial_sidebar_state="expanded"
)

st.header('Aspect Based Sentiment Analysis (ABSA) System')

st.write(f"*Simply download the Template CSV file, change the Time Period and Reviews of interest, and upload the modified CSV file.*")
st.write(f"*ABSA models will then run in the background (the models run time can be anywhere between few seconds to hours, depending on the size of the modified CSV file).*")
st.write(f"*The results will be displayed after the models have finished running. The output CSV results can also be downloaded.*")

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

st.subheader('Download your Template CSV file here:')

st.download_button("Download Template CSV file",
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

    st.subheader('Results')

    col1, col2 = st.columns([1,1])

    with col1:
        st.write("Placeholder1 (Wordcloud Visualization)") # TODO insert word-cloud
    with col2:
        st.write("Placeholder2 (Treemap Visualization)")

        # Tree Map
        timePeriod_str = rawInput_df.head(1).reset_index()['Time period'][0]

        # Normalize sentiment values and apply colors
        norm = matplotlib.colors.Normalize(vmin=min(overallResultsOutput_df.Sentiment), vmax=max(overallResultsOutput_df.Sentiment))
        colors = [matplotlib.cm.Reds(norm(value)) for value in overallResultsOutput_df.Sentiment]

        # Create figure and size
        fig, ax = plt.subplots()
        fig.set_size_inches(12, 8)

        # Create squarify plot
        squarify.plot(
            label=overallResultsOutput_df.Category,
            sizes=overallResultsOutput_df.Total,
            value=overallResultsOutput_df.Sentiment,
            color=colors,
            alpha=.5,
            pad=True
        )

        # Add titles
        plt.suptitle("Sentiment Heat Map of Printer & Ink Aspects", fontsize=20, fontweight="bold")  # Main title
        plt.title(f"{timePeriod_str}", fontsize=14, fontstyle='italic', pad=10)  # Subtitle

        # Remove axes
        plt.axis('off')

        # Display the plot in Streamlit
        st.pyplot(fig)

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