import pandas as pd
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize
from transformers import pipeline
# import model from hugging face

def aspectClassification(self, rawInput_file):

    df = rawInput_file

    # Pre-processing steps to strip each review into multiple sentences

    def split_into_sentences(df):
        rows = []
        
        for idx, row in df.iterrows():
            sentences = sent_tokenize(row['Reviews'])  # Split the review into sentences
            for sentence in sentences:
                rows.append({'Review_ID': idx, 'Sentence': sentence})  # Keep track of Review ID
                
        return pd.DataFrame(rows)

    # Apply the function to your rawInput_df
    sentence_df = split_into_sentences(df)

    # Run classification model for each sentence

    # Load pre-trained zero-shot classification model for aspect classification
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    # Aspects you're classifying
    aspects = ['Price', 'Customer Service', 'Product Quality', 'Delivery']

    def classify_aspect(row, aspects):
        # Classify aspect using the provided classifier
        aspect_result = classifier(row, aspects)
        aspect = aspect_result['labels'][0]  # The most probable aspect
        return aspect

    # Apply the aspect classification
    sentence_df['Aspect'] = sentence_df['Sentence'].apply(lambda row: classify_aspect(row, aspects))

    aspectOutput_df = sentence_df

    # Transfrom df to aspectOutput_df and replace hard-coding below
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
    #     ]
    # }
    # aspectOutput_df = pd.DataFrame(data)

    return aspectOutput_df