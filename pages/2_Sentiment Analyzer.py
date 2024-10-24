import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import squarify
import string
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
from controller import controllerService

# download NLTK packages
nltk.download('stopwords')
nltk.download('wordnet')

# Set page config
st.set_page_config(
    page_title='Sentiment Analyzer', # Set display name of browser tab
    layout="wide", # "wide" or "centered"
    initial_sidebar_state="expanded"
)

st.header('Aspect Based Sentiment Analysis (ABSA) System')

st.write(f"*Simply download the Template CSV file, change the Time Period and Reviews of interest, and upload the modified CSV file.*")
st.write(f"*ABSA models will then run in the background (the models run time can be anywhere between few seconds to hours, depending on the size of the modified CSV file).*")
st.write(f"*The results will be displayed after the models have finished running. The output CSV files can also be downloaded.*")

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

    if not uploaded_file.name.endswith('.csv'):
        st.write('Uploaded file is not in CSV format. Please try again.')

    # Attempt to read the uploaded file as a CSV
    rawInput_df = pd.read_csv(uploaded_file)
    
    # Check if the required columns are present
    if {'Time period', 'Reviews'}.issubset(rawInput_df.columns):
        if rawInput_df[['Time period', 'Reviews']].empty:
            st.write("The uploaded CSV file is empty. Please provide data.")
        else:
            # Process the DataFrame as needed
            st.write("Data loaded successfully.")

            modelResults = controllerService()
            aspect_df = modelResults.runAspectClassification(rawInput_df)
            aspectSentimentOutput_df, overallResultsOutput_df = modelResults.runsentimentAnalyzer(aspect_df)

            timePeriod_str = rawInput_df.head(1).reset_index()['Time period'][0]

            st.subheader('Results')
            st.write(f"Time period: *{timePeriod_str}*")
            if overallResultsOutput_df.empty:
                    st.write('There are neither Positive nor Negative reviews. Heat Map will not be generated. CSV files will not be available for download.')

            col1, col2 = st.columns([1,1])

            with col1:

                def preprocess_text(tokens):

                    # Convert all characters to lower case
                    tokens = [t.lower() for t in tokens]

                    # Remove Punctuations
                    tokens = [t for t in tokens if t not in string.punctuation]

                    # Remove Stopwords
                    stop = stopwords.words('english')
                    tokens = [t for t in tokens if t not in stop]

                    # Remove from filtered list (additional)
                    filter_list = ["“", "”", "would", "could", "'s", "left", "right", "a.m.", "p.m."]
                    tokens = [t for t in tokens if t not in filter_list]

                    # Remove Numbers/Numerics
                    tokens = [t for t in tokens if not t.isnumeric()]

                    # Lemmatization
                    wnl = nltk.WordNetLemmatizer()
                    tokens = [wnl.lemmatize(t) for t in tokens] 

                    return tokens
                
                # Apply the preprocessing to each review
                rawInput_df['Processed_Reviews'] = rawInput_df['Reviews'].apply(lambda x: ' '.join(preprocess_text(x.split())))

                # Combine all reviews into a single string for word cloud
                all_reviews = ' '.join(rawInput_df['Processed_Reviews'])
                
                if all_reviews != '':

                    # Generate the word cloud
                    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_reviews)

                    # Plot the word cloud using Streamlit
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    plt.figure(figsize=(12, 8))
                    plt.suptitle("Word Cloud of Reviews", fontsize=20, fontweight='bold') # Main title
                    plt.imshow(wordcloud, interpolation='bilinear')
                    plt.axis("off")

                    # Display the word cloud in Streamlit
                    st.pyplot(plt)
                
                else:
                    st.write('Reviews do not meet requirements for Word Cloud generation.')
                
            with col2:
                
                if not overallResultsOutput_df.empty:

                    # Tree Map

                    # Normalize sentiment values and apply colors
                    norm = matplotlib.colors.Normalize(vmin=min(overallResultsOutput_df.Sentiment), vmax=max(overallResultsOutput_df.Sentiment))
                    colors = [matplotlib.cm.Reds_r(norm(value)) for value in overallResultsOutput_df.Sentiment]
                    # # Create a colormap from red to green
                    # cmap = plt.get_cmap('RdYlGn')  # Use a diverging color map (Red to Yellow to Green)
                    # colors = [cmap(norm(value)) for value in overallResultsOutput_df.Sentiment]

                    # Create figure and size
                    fig, ax = plt.subplots()
                    fig.set_size_inches(12, 8)

                    # Create squarify plot
                    squarify.plot(
                        label=overallResultsOutput_df.Category,
                        sizes=overallResultsOutput_df.Total,
                        value=overallResultsOutput_df.Sentiment,
                        color=colors,
                        alpha=.6,
                        pad=True
                    )

                    # Add title
                    plt.suptitle("Sentiment Heat Map of Printer & Ink Aspects", fontsize=20, fontweight="bold")  # Main title

                    # Remove axes
                    plt.axis('off')

                    # Display the plot in Streamlit
                    st.pyplot(fig)

                    st.write("*The Sentiment Score for each aspect is normalized between -1 to 1 (-1 = Worse, 0 = Neutral, 1 = Best)*")

            if not overallResultsOutput_df.empty:

                st.download_button("Download Aspect-Sentiment Output CSV file",
                                aspectSentimentOutput_df.to_csv(index = False),
                                file_name = 'aspectSentimentOutput_file.csv',
                                mime = 'text/csv')
                
                st.download_button("Download Overall Results Output CSV file",
                                overallResultsOutput_df.to_csv(index = False),
                                file_name = 'overallResultsOutput_file.csv',
                                mime = 'text/csv')
                
    else:
        st.write("Uploaded CSV file does not contain the required columns. Please label the columns as 'Time period' and 'Reviews'.")

except:
    pass