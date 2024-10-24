import pandas as pd
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import ast

def sentimentAnalyzer(self, aspectInput_df):

    #####################################################################################################

    # TO UNCOMMENT (for production)

    chosenModel = 'facebook/bart-large-mnli' # 'facebook/bart-large-mnli' # MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli

    df = aspectInput_df.copy()

    def initialize_nli_model(model_name=chosenModel):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        
        # Create a pipeline for NLI, specifying the device
        nli_pipeline = pipeline("zero-shot-classification", model=model, tokenizer=tokenizer, device=device)
        
        return nli_pipeline, device

    # Function to extract sentiment expressions
    def extract_sentiment_expression_nli(review, aspects, nli_pipeline):
        # Define possible labels for sentiment
        candidate_labels = ['positive', 'negative']

        # Store the answers
        answers = {}

        # Iterate over the provided aspects to construct the NLI inputs
        for aspect in aspects:
            # Formulate the hypothesis
            hypothesis = f"The sentiment for the aspect '{aspect}' is"

            # Use the NLI pipeline to predict the sentiment for each aspect
            response = nli_pipeline(
                sequences=review,   # Premise: The review text
                candidate_labels=[f"{hypothesis} {label}" for label in candidate_labels],  # Hypotheses
            )

            # Extract the sentiment with the highest score
            sentiment = response['labels'][0].split()[-1]  # Get the last word ('positive' or 'negative')
            answers[aspect] = sentiment

        return answers

    # Function to convert 'Predicted Labels' column to a list
    def convert_to_list(aspect_string):
         return ast.literal_eval(aspect_string) if isinstance(aspect_string, str) else aspect_string

    # Applying the function to the DataFrame
    def process_dataframe(df, nli_pipeline):
        # Create a new column for the list of aspects
        df['Aspect List'] = df['Predicted Labels'].apply(convert_to_list)

        # Apply the sentiment extraction function to each row, passing nli_pipeline
        df['Sentiment Expressions'] = df.apply(lambda row: extract_sentiment_expression_nli(row['Reviews'], row['Aspect List'], nli_pipeline), axis=1)
        
        return df

    # Initialize the model and process the dataframe
    nli_pipeline, device = initialize_nli_model(chosenModel)
    
    # Apply to df
    df_new = process_dataframe(df, nli_pipeline)

    # Function to process the sentiment expression and assign the label
    def process_sentiment_label(sentiment):
         if any(keyword in sentiment for keyword in ['negative']):
             return 'Negative'
         elif any(keyword in sentiment for keyword in ['positive']):
             return 'Positive'
         else:
             return 'Neutral'

    # Function to expand rows for each aspect and sentiment
    def expand_rows_for_aspects(df):
         expanded_rows = []

         # Iterate through each row in the DataFrame
         for index, row in df.iterrows():
             # Get the sentiment dictionary from the row
             sentiments = row['Sentiment Expressions']  # This is a dict 

             # Check if sentiments is a dictionary and not empty
             if isinstance(sentiments, dict) and sentiments:
                 # Iterate over each aspect in the sentiment dictionary
                 for aspect, sentiment_expression in sentiments.items():
                     new_row = row.copy()  # Copy the current row
                    
                     # Create a new column for the current aspect
                     new_row['Aspect'] = aspect
                    
                     # Create a new column for the sentiment label based on the sentiment expression
                     new_row['Sentiment'] = process_sentiment_label(sentiment_expression)
                    
                     # Append the new row to the list
                     expanded_rows.append(new_row)
             else:
                 # If there are no sentiments, append the original row without modifications
                 expanded_rows.append(row)

         # Create a new DataFrame from the expanded rows
         expanded_df = pd.DataFrame(expanded_rows)
        
        #  # Filter out rows where 'Sentiment' is empty
        #  expanded_df = expanded_df[(expanded_df['Sentiment'].notnull()) | (expanded_df['Sentiment'] != '')]
        
         return expanded_df

    aspectSentimentOutput_df = expand_rows_for_aspects(df_new)
    aspectSentimentOutput_df.drop(columns = ['Predicted Labels', 'Aspect List', 'Sentiment Expressions'], inplace = True)
    aspectSentimentOutput_df['Time period'] = aspectSentimentOutput_df['Time period'].where(
                                            aspectSentimentOutput_df['Time period'] != aspectSentimentOutput_df['Time period'].shift())

    df = aspectSentimentOutput_df.copy()

    #####################################################################################################

    # TO COMMENT (just for system demo)

    # sentiment_list = [
    #     "Positive", "Positive", "Negative", "Positive", "Positive", "Positive", "Positive", 
    #     "Negative", "Positive", "Negative", "Positive", "Positive", "Positive", "Negative", 
    #     "Positive", "Positive", "Negative", "Positive", "Negative", "Positive", "Positive", 
    #     "Positive", "Negative", "Positive", "Negative", "Negative", "Positive", "Positive", 
    #     "Negative", "Negative", "Negative", "Positive", "Negative", "Negative", "Negative", 
    #     "Negative", "Negative", "Negative", "Negative", "Negative", "Negative", "Positive", 
    #     "Positive", "Positive", "Negative", "Positive", "Negative", "Positive", "Negative", 
    #     "Positive", "Positive", "Positive", "Positive", "Negative", "Negative", "Positive", 
    #     "Negative", "Negative", "Negative", "Negative", "Positive", "Positive", "Negative", 
    #     "Negative", "Positive", "Positive", "Negative", "Negative", "Negative", "Positive", 
    #     "Positive", "Positive", "Negative", "Positive"
    # ]
    # aspectInput_df['Sentiment'] = sentiment_list
    # aspectSentimentOutput_df = aspectInput_df.copy()
    # df = aspectSentimentOutput_df.copy()

    #####################################################################################################
    
    # Transfrom df

    try:

        # Create a pivot table to count positive and negative sentiments for each aspect
        df = df.pivot_table(index='Aspect', columns='Sentiment', aggfunc='size', fill_value=0)
        # Add a 'Total' column to get the sum of positive and negative sentiments
        df['Total'] = df['Positive'] + df['Negative']
        # Represent overall sentiment for the category based on no. of 'Positive'/'Negative'
        df["Sentiment"] = np.round((df["Positive"] - df["Negative"]) / (df["Positive"] + df["Negative"]), 2)
        # Set 'Category' column as index
        df["Category"] = df.index

    except KeyError:

        if 'Aspect' in df.columns and 'Sentiment' in df.columns: # there is either Positive or Negative comments

            if 'Positive' in df.columns and 'Negative' not in df.columns:
                # Create a pivot table to count positive and negative sentiments for each aspect
                df = df.pivot_table(index='Aspect', columns='Sentiment', aggfunc='size', fill_value=0)
                df['Total'] = df['Positive']
                df["Sentiment"] = 1
                # Set 'Category' column as index
                df["Category"] = df.index

            elif 'Negative' in df.columns and 'Positive' not in df.columns:
                # Create a pivot table to count positive and negative sentiments for each aspect
                df = df.pivot_table(index='Aspect', columns='Sentiment', aggfunc='size', fill_value=0)
                df['Total'] = df['Negative']
                df["Sentiment"] = -1
                # Set 'Category' column as index
                df["Category"] = df.index

        else: # there are no sentiments extracted
            overallResultsOutput_df = pd.DataFrame()  
    
    return aspectSentimentOutput_df, overallResultsOutput_df # aspect-sentiment result outputs; aggregated final outputs
