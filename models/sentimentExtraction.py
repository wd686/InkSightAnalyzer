import pandas as pd
import numpy as np
from transformers import pipeline
# import model from hugging face

def sentimentExtraction(self, aspectInput_df):

    #####################################################################################################

    # TODO TO UNCOMMENT

    # # Load sentiment analysis model
    # sentiment_model = pipeline('sentiment-analysis') # 'distilbert-base-uncased-finetuned-sst-2-english' model

    # df = aspectInput_df.copy()

    # # Run sentiment model for each sentence (containing 1 aspect) to obtain sentiment

    # def classify_sentiment(row):
    #     # Classify sentiment using the sentiment model
    #     sentiment_result = sentiment_model(row)
    #     sentiment = sentiment_result[0]['label']
    #     return sentiment
    
    # # Apply sentiment classification and append the result
    # df['Sentiment'] = df['Sentence'].apply(classify_sentiment)

    # df.loc[df.Sentiment == 'POSITIVE', 'Sentiment'] = 'Positive'
    # df.loc[df.Sentiment == 'NEGATIVE', 'Sentiment'] = 'Negative'
    # aspectSentimentOutput_df = df.copy()

    #####################################################################################################

    # TODO TO COMMENT

    sentiment_list = [
        "Positive", "Positive", "Negative", "Positive", "Positive", "Positive", "Positive", 
        "Negative", "Positive", "Negative", "Positive", "Positive", "Positive", "Negative", 
        "Positive", "Positive", "Negative", "Positive", "Negative", "Positive", "Positive", 
        "Positive", "Negative", "Positive", "Negative", "Negative", "Positive", "Positive", 
        "Negative", "Negative", "Negative", "Positive", "Negative", "Negative", "Negative", 
        "Negative", "Negative", "Negative", "Negative", "Negative", "Negative", "Positive", 
        "Positive", "Positive", "Negative", "Positive", "Negative", "Positive", "Negative", 
        "Positive", "Positive", "Positive", "Positive", "Negative", "Negative", "Positive", 
        "Negative", "Negative", "Negative", "Negative", "Positive", "Positive", "Negative", 
        "Negative", "Positive", "Positive", "Negative", "Negative", "Negative", "Positive", 
        "Positive", "Positive", "Negative", "Positive"
    ]
    aspectInput_df['Sentiment'] = sentiment_list
    aspectSentimentOutput_df = aspectInput_df.copy()
    df = aspectSentimentOutput_df.copy()

    #####################################################################################################
        
    # Transfrom df to aspectSentimentOutput_df

    df.drop(columns = ['Sentence', 'Review_ID'], inplace = True)

    # Create a pivot table to count positive and negative sentiments for each aspect
    df = df.pivot_table(index='Aspect', columns='Sentiment', aggfunc='size', fill_value=0)

    #Add a 'Total' column to get the sum of positive and negative sentiments
    df['Total'] = df['Positive'] + df['Negative']

     #add two new columns into the table
    df["Category"] = df.index

    #represent overall sentiment for the categary based on num of pos/neg
    df["Sentiment"] = np.round((df["Positive"]-df["Negative"])/
                                                     (df["Negative"]+df["Positive"]),2)
    
    overallResultsOutput_df = df.copy()
    
    return aspectSentimentOutput_df, overallResultsOutput_df # aspect-sentiment result outputs; aggregated final outputs