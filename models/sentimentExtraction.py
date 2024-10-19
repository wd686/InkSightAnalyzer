import pandas as pd
import numpy as np
from transformers import pipeline
# import model from hugging face

# hardcoded variables
## ...

def sentimentExtraction(self, aspectInput_df):

    df = aspectInput_df

    # Run sentiment model for each sentence (containing 1 aspect) to obtain sentiment

    # Load sentiment analysis model
    sentiment_model = pipeline('sentiment-analysis')

    def classify_sentiment(row):
        # Classify sentiment using the sentiment model
        sentiment_result = sentiment_model(row)
        sentiment = sentiment_result[0]['label']
        return sentiment
    
    # Apply sentiment classification and append the result
    df['Sentiment'] = df['Sentence'].apply(classify_sentiment)

    aspectSentimentOutput_df = df

    # Transfrom df to aspectSentimentOutput_df

    aspectSentimentOutput_df.drop(columns = ['Sentence', 'Review_ID'], inplace = True)

    # Create a pivot table to count positive and negative sentiments for each aspect
    aspectSentimentOutput_df = aspectSentimentOutput_df.pivot_table(index='Aspect', columns='Sentiment', aggfunc='size', fill_value=0)

    ###########
    # Add a 'Total' column to get the sum of positive and negative sentiments
    aspectSentimentOutput_df['Total'] = aspectSentimentOutput_df['Positive'] + aspectSentimentOutput_df['Negative']

    #add two new columns into the table
    # aspectSentimentOutput_df["Category"] = aspectSentimentOutput_df.index

    # #represent overall sentiment for the categary based on num of pos/neg
    # aspectSentimentOutput_df["Sentiment"] = np.round((aspectSentimentOutput_df["Positive"]-aspectSentimentOutput_df["Negative"])/
    #                                                  (aspectSentimentOutput_df["Negative"]+aspectSentimentOutput_df["Positive"]),2)
    
    # aspectSentimentOutput_df.rename(columns = {'Positive':'Pos', 'Negative':'Neg'}, inplace = True)
    #######################

    overallResultsOutput_df = aspectSentimentOutput_df

    # data = {
    #     'Reviews': [
    #         'This printer sucks. I want a refund.',
    #         'This printer sucks. I want a refund.'
    #     ],
    #     'Sentence': [
    #         'This printer sucks.',
    #         'I want a refund.'
    #     ],
    #     'Aspect': [
    #         'Quality',
    #         'Cost'
    #     ],
    #     'Sentiment': [
    #         'Negative',
    #         'Negative'
    #     ]
    # }

    # aspectSentimentOutput_df = pd.DataFrame(data)
        
    # Aggregate results into final results output
    # Transfrom aspectSentimentOutput_df to overallResultsOutput_df and replace hard-coding below

    # data = {
    #     'Pos': [4, 33, 42, 0, 22],
    #     'Neg': [54, 0, 3, 47, 22],
    #     'Total': [58, 33, 45, 47, 44],
    #     'Category': ['Product Quality', 'Price', 'Delivery', 'Customer Service', 'Instant Ink'],
    #     'Sentiment': [-0.86, 1.00, 0.87, -1.00, 0.00]
    # }

    # data = {
    #     'Pos': [4, 17, 42, 16],
    #     'Neg': [54, 14, 3, 25],
    #     'Total': [58, 31, 45, 41],
    #     'Category': ['Price', 'Customer Service', 'Product Quality', 'Delivery'],
    #     'Sentiment': [-0.86, 0.10, 0.87, -0.22]
    # }

    # overallResultsOutput_df = pd.DataFrame(data)
    
    return aspectSentimentOutput_df, overallResultsOutput_df # aspect-sentiment result outputs; aggregated final outputs