import pandas as pd
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import ast

def sentimentAnalyzer(self, aspectInput_df):

    chosenModel = 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'

    df = aspectInput_df.copy()

    # aspectSentimentOutput_df = aspectInput_df.copy()

    def initialize_nli_model(model_name=chosenModel):
        # device = 0 if torch.cuda.is_available() else -1  # Check if GPU is available
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
        
         # Filter out rows where 'Predicted Sentiment' is blank or NaN
         expanded_df = expanded_df[expanded_df['Sentiment'].notna() | (expanded_df['Sentiment'] != '')]
        
         return expanded_df

    aspectSentimentOutput_df = expand_rows_for_aspects(df_new)
    aspectSentimentOutput_df.drop(columns = ['Predicted Labels', 'Aspect List', 'Sentiment Expressions'], inplace = True)
    aspectSentimentOutput_df['Time period'] = aspectSentimentOutput_df['Time period'].where(
                                            aspectSentimentOutput_df['Time period'] != aspectSentimentOutput_df['Time period'].shift())

    df = aspectSentimentOutput_df.copy()
    
    #################################################################################################### 
    # Transfrom df

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

    # # TODO TO DELETE!
    # data = {
    #     'Positive': [4, 17, 42, 16],
    #     'Negative': [54, 14, 3, 25],
    #     'Total': [58, 31, 45, 41],
    #     'Category': ['Price', 'Customer Service', 'Product Quality', 'Delivery'],
    #     'Sentiment': [-0.86, 0.10, 0.87, -0.22]
    # }
    # overallResultsOutput_df = pd.DataFrame(data)
    
    return aspectSentimentOutput_df, overallResultsOutput_df # aspect-sentiment result outputs; aggregated final outputs


























# import pandas as pd
# import numpy as np
# import ollama
# import ast
# from transformers import pipeline

# def sentimentAnalyzer(self, aspectInput_df):

#     #####################################################################################################

#     # TODO TO UNCOMMENT

#     # # Load sentiment analysis model
#     # sentiment_model = pipeline('sentiment-analysis') # 'distilbert-base-uncased-finetuned-sst-2-english' model

#     # df = aspectInput_df.copy()

#     # # Run sentiment model for each sentence (containing 1 aspect) to obtain sentiment

#     # def classify_sentiment(row):
#     #     # Classify sentiment using the sentiment model
#     #     sentiment_result = sentiment_model(row)
#     #     sentiment = sentiment_result[0]['label']
#     #     return sentiment
    
#     # # Apply sentiment classification and append the result
#     # df['Sentiment'] = df['Sentence'].apply(classify_sentiment)

#     # df.loc[df.Sentiment == 'POSITIVE', 'Sentiment'] = 'Positive'
#     # df.loc[df.Sentiment == 'NEGATIVE', 'Sentiment'] = 'Negative'
#     # aspectSentimentOutput_df = df.copy()

#     #####################################################################################################

#     # TODO TO COMMENT

#     sentiment_list = [
#         "Positive", "Positive", "Negative", "Positive", "Positive", "Positive", "Positive", 
#         "Negative", "Positive", "Negative", "Positive", "Positive", "Positive", "Negative", 
#         "Positive", "Positive", "Negative", "Positive", "Negative", "Positive", "Positive", 
#         "Positive", "Negative", "Positive", "Negative", "Negative", "Positive", "Positive", 
#         "Negative", "Negative", "Negative", "Positive", "Negative", "Negative", "Negative", 
#         "Negative", "Negative", "Negative", "Negative", "Negative", "Negative", "Positive", 
#         "Positive", "Positive", "Negative", "Positive", "Negative", "Positive", "Negative", 
#         "Positive", "Positive", "Positive", "Positive", "Negative", "Negative", "Positive", 
#         "Negative", "Negative", "Negative", "Negative", "Positive", "Positive", "Negative", 
#         "Negative", "Positive", "Positive", "Negative", "Negative", "Negative", "Positive", 
#         "Positive", "Positive", "Negative", "Positive"
#     ]
#     aspectInput_df['Sentiment'] = sentiment_list
#     aspectSentimentOutput_df = aspectInput_df.copy()
#     df = aspectSentimentOutput_df.copy()

#     #####################################################################################################

#     # df = aspectInput_df.copy()

#     # # Function to run llama 3.1
#     # def extract_sentiment_expression_llama (review, aspects, model_name='llama3.1'):
#     #     # Store the answers
#     #     answers = {}

#     #     # Iterate over the provided aspects to construct the prompt
#     #     for aspect in aspects:
#     #         prompt = f"""
#     #         Review: "{review}"
#     #         Aspect: "{aspect}"
#     #         What is the sentiment (positive, negative) for this aspect? Return the sentiment identified only. 
#     #         """

#     #         # Use the Ollama API to generate the sentiment expression
#     #         response = ollama.chat(
#     #             model=model_name, 
#     #             messages=[{"role": "user", "content": prompt}]
#     #         )

#     #         # Extract the sentiment expression from the response
#     #         result_text = response['message']['content']
#     #         answers[aspect] = result_text.strip()

#     #     return answers


#     # # Function to convert 'Predicted Labels' column to a list
#     # def convert_to_list(aspect_string):
#     #     return ast.literal_eval(aspect_string) if isinstance(aspect_string, str) else aspect_string

#     # # Applying the function to the DataFrame
#     # def process_dataframe(df):
#     #     # Create a new column for the list of aspects
#     #     df['Aspect List'] = df['Predicted Labels'].apply(convert_to_list)

#     #     # Apply the sentiment extraction function to each row
#     #     df['Sentiment Expressions'] = df.apply(
#     #     lambda row: extract_sentiment_expression_llama(row['Reviews'], row['Aspect List']), axis=1
#     # )

#     #     return df


#     # # apply to df
#     # df_new = process_dataframe(df)


#     # # Function to process the sentiment expression and assign the label
#     # def process_sentiment_label(sentiment):
#     #     if any(keyword in sentiment for keyword in ['Negative', 'Concern', 'Not Satisfied', 'Mixed']):
#     #         return 'Negative'
#     #     elif any(keyword in sentiment for keyword in ['Positive', 'Satisfied']):
#     #         return 'Positive'
#     #     else:
#     #         return 'Neutral'

#     # # Function to expand rows for each aspect and sentiment
#     # def expand_rows_for_aspects(df):
#     #     expanded_rows = []

#     #     # Iterate through each row in the DataFrame
#     #     for index, row in df.iterrows():
#     #         # Get the sentiment dictionary from the row
#     #         sentiments = row['Sentiment Expressions']  # This is a dict 

#     #         # Check if sentiments is a dictionary and not empty
#     #         if isinstance(sentiments, dict) and sentiments:
#     #             # Iterate over each aspect in the sentiment dictionary
#     #             for aspect, sentiment_expression in sentiments.items():
#     #                 new_row = row.copy()  # Copy the current row
                    
#     #                 # Create a new column for the current aspect
#     #                 new_row['Aspect'] = aspect
                    
#     #                 # Create a new column for the sentiment label based on the sentiment expression
#     #                 new_row['Predicted Sentiment'] = process_sentiment_label(sentiment_expression)
                    
#     #                 # Append the new row to the list
#     #                 expanded_rows.append(new_row)
#     #         else:
#     #             # If there are no sentiments, append the original row without modifications
#     #             expanded_rows.append(row)

#     #     # Create a new DataFrame from the expanded rows
#     #     expanded_df = pd.DataFrame(expanded_rows)
        
#     #     # Filter out rows where 'Predicted Sentiment' is blank or NaN
#     #     # expanded_df = expanded_df[expanded_df['Predicted Sentiment'].notna() & (expanded_df['Predicted Sentiment'] != '')] # TODO uncomment?
        
#     #     return expanded_df


#     # aspectSentimentOutput_df = expand_rows_for_aspects(df_new)
#     # df = aspectSentimentOutput_df.copy()
    
#     ##################################################################################################### 
#     # Transfrom df

#     df.drop(columns = ['Sentence', 'Review_ID'], inplace = True)

#     # Create a pivot table to count positive and negative sentiments for each aspect
#     df = df.pivot_table(index='Aspect', columns='Sentiment', aggfunc='size', fill_value=0)
#     # df = df.pivot_table(index='Aspect', columns='Predicted Sentiment', aggfunc='size', fill_value=0)

#     #Add a 'Total' column to get the sum of positive and negative sentiments
#     df['Total'] = df['Positive'] + df['Negative']

#      #add two new columns into the table
#     df["Category"] = df.index

#     #represent overall sentiment for the categary based on num of pos/neg
#     df["Sentiment"] = np.round((df["Positive"]-df["Negative"])/
#                                                      (df["Negative"]+df["Positive"]),2)
    
#     overallResultsOutput_df = df.copy()

#     # # TODO TO DELETE!
#     # data = {
#     #     'Positive': [4, 17, 42, 16],
#     #     'Negative': [54, 14, 3, 25],
#     #     'Total': [58, 31, 45, 41],
#     #     'Category': ['Price', 'Customer Service', 'Product Quality', 'Delivery'],
#     #     'Sentiment': [-0.86, 0.10, 0.87, -0.22]
#     # }
#     # overallResultsOutput_df = pd.DataFrame(data)
    
#     return aspectSentimentOutput_df, overallResultsOutput_df # aspect-sentiment result outputs; aggregated final outputs