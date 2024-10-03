import pandas as pd
import numpy as np
import nltk
import warnings
from transformers import pipeline
from nltk.tokenize import sent_tokenize
warnings.filterwarnings("ignore", category=FutureWarning)

# hard-codings

monthsOfInterest_list = ["2023-12-01", "2024-01-01"] # "2023-12-01", "2024-01-01", "2024-04-01"
model = "facebook/bart-large-mnli" # "facebook/bart-large-mnli", "cross-encoder/nli-roberta-base", "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
input_filepath = "combined_df.csv"
output_filepath = f"nli_combined_df_{model.replace('/', '-')} ({', '.join(monthsOfInterest_list)}).csv"
maxTokenCount = 200
sample = 10

# functions

# def get_sentiment_labels(text):
#     if isinstance(text, str):  # Check if the input is a valid string
#         # Get results from each pipeline
#         result_sa1 = sa1(text)[0]
#         result_sa2 = sa2(text)[0]
#         result_sa3 = sa3(text)[0]
        
#         # Return the labels with the highest scores
#         label_sa1 = result_sa1['label']
#         label_sa2 = result_sa2['label']
#         label_sa3 = result_sa3['label']
        
#         return pd.Series([label_sa1, label_sa2, label_sa3])
#     else:
#         # Return 'Unknown' for non-string inputs
#         return pd.Series(['Unknown', 'Unknown', 'Unknown'])
    
# Define a function to extract NLI labels for all three models
def get_nli_label(text):
    if isinstance(text, str):  # Check if the input is a valid string
        # Perform zero-shot classification with candidate labels for each model
        # Specify candidate labels for nli
        candidate_labels = ["positive comment", "negative comment", "neutral comment"]
        result_nli = nli(text, candidate_labels=candidate_labels)

        # Extract the label with the highest probability from each result
        label_nli = result_nli['labels'][0]  # First label is the highest probability
        
        return pd.Series([label_nli])
    else:
        # Return 'Unknown' for non-string inputs
        return pd.Series(['Unknown'])
    
############################################################

### WRANGLING ###

# Read the csv files into DataFrames
combined_df = pd.read_csv(input_filepath) 

# Filter recent reviews
# combined_df['Month of Response Date'] = pd.to_datetime(combined_df['Month of Response Date'])
pattern = '|'.join(monthsOfInterest_list)
combined_df = combined_df[combined_df['Month of Response Date'].str.contains(pattern, na=False)].reset_index(drop=True)

# # combined_df.head(3)
# len(combined_df)

# df = combined_df

# # Count the frequency of each token count
# token_count_summary = df['token_count'].value_counts().sort_index()

# # Convert to DataFrame for better readability
# token_count_summary_df = token_count_summary.reset_index()
# token_count_summary_df.columns = ['Token Count', 'Row Count']
# token_count_summary_df = token_count_summary_df[token_count_summary_df['Token Count'] >= 512]

# token_count_summary_df.sort_values(by='Token Count', ascending=False).reset_index(drop=True)

############################################################

### LOAD PIPELINE ###

# load the pipeline and models

# pipeline for Sentiment Analysis (SA)
# sa1: distilbert/distilbert-base-uncased-finetuned-sst-2-english -> # https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english?library=transformers
# sa2: siebert/sentiment-roberta-large-english -> # https://huggingface.co/siebert/sentiment-roberta-large-english?library=transformers
# sa3: avichr/heBERT_sentiment_analysis -> # https://huggingface.co/avichr/heBERT_sentiment_analysis?library=transformers

# sa1 = pipeline("text-classification", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english", truncation=True)
# sa2 = pipeline("text-classification", model="siebert/sentiment-roberta-large-english", truncation=True)
# sa3 = pipeline("text-classification", model="avichr/heBERT_sentiment_analysis", truncation=True)

# pipeline for Extractive Question-Answering (QA)
# qa1: deepset/roberta-base-squad2 -> # https://huggingface.co/deepset/roberta-base-squad2?library=transformers
# qa2: distilbert/distilbert-base-cased-distilled-squad -> # https://huggingface.co/distilbert/distilbert-base-cased-distilled-squad?library=transformers
# qa3: google-bert/bert-large-uncased-whole-word-masking-finetuned-squad -> # https://huggingface.co/google-bert/bert-large-uncased-whole-word-masking-finetuned-squad?library=transformers

# qa1 = pipeline("question-answering", model="deepset/roberta-base-squad2")       
# qa2 = pipeline("question-answering", model="distilbert/distilbert-base-cased-distilled-squad")    
# qa3 = pipeline("question-answering", model="google-bert/bert-large-uncased-whole-word-masking-finetuned-squad")

# pipeline for Zero-Shot Classification w NLI (NLI), better to identify keywords for candidate_labels
# nli1: facebook/bart-large-mnli -> # https://huggingface.co/facebook/bart-large-mnli?library=transformers
# nli2: MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli -> # https://huggingface.co/MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli?library=transformers
# nli3: cross-encoder/nli-roberta-base -> # https://huggingface.co/cross-encoder/nli-roberta-base?library=transformers

nli = pipeline("zero-shot-classification", model = model)

# Specify candidate labels for nli
# candidate_labels = ["positive comment", "negative comment", "neutral comment"]
                                                     
# text = "This movie was absolutely amazing!"
# text = combined_df['Combined Text'][1]

# print(f"Text: {text}\n\n"
#       f"sa1: {sa1(text, truncation=True, max_length=512)}\n"
#       f"sa2: {sa2(text, truncation=True, max_length=512)}\n"
#       f"sa3: {sa3(text, truncation=True, max_length=512)}\n\n"
#     #   f"qa1: {qa1(question='What is this comment about?', context=text, truncation=True, max_length=512)}\n"
#     #   f"qa2: {qa2(question='What is this comment about?', context=text, truncation=True, max_length=512)}\n"
#     #   f"qa3: {qa3(question='What is this comment about?', context=text, truncation=True, max_length=512)}\n\n"
#       f"nli1: {nli1(text, candidate_labels=candidate_labels, truncation=True, max_length=512)}\n"
#       f"nli1: {nli2(text, candidate_labels=candidate_labels, truncation=True, max_length=512)}\n"
#       f"nli1: {nli3(text, candidate_labels=candidate_labels, truncation=True, max_length=512)}\n"
#       )

############################################################

sa_combined_df = combined_df[combined_df['token_count'] <= maxTokenCount].copy()

# Apply the function to each row in the 'Combined Text' column and create new columns
# sa_combined_df[['sa1_label', 'sa2_label', 'sa3_label']] = sa_combined_df['Combined Text'].apply(get_sentiment_labels)
sa_combined_df[['nli_label']] = sa_combined_df['Combined Text'].apply(get_nli_label)

############################################################

sa_combined_df.to_csv(output_filepath)

### Multi-turn Zero-shot ABSA ###

# # 50 rows -> ~1min
# reviews = combined_df['Combined Text'].sample(sample)

# nli = nli1

# # Define categories
# cats = ["printer", "ink", "service", "delivery", "general"]

# # Initialize result list and sentence count
# results = []
# scount = 0

# # Process each review
# for r in reviews:
#     scount += 1
#     s = r  # Since each review is already one sentence, no need for tokenization
#     labels = []

#     # Find applicable categories
#     for c in cats:
#         yes = f"It's a comment on {c}"
#         no = f"It's not a comment on {c}"
#         res = nli(s, candidate_labels=[yes, no])
#         label = res['labels'][0]
#         if res['scores'][0] >= 0.9 and "not" not in label:
#             labels.append(c)

#     # If no specific category is detected, assign "general"
#     if not labels:
#         labels.append('general')

#     # Determine the polarity for each detected category
#     for l in labels:
#         if l == "general":
#             pos = "This is a positive comment in general"
#             neg = "This is a negative comment in general"
#             neu = "This is a neutral comment in general"
#         else:
#             pos = f"This is a positive comment on {l}"
#             neg = f"This is a negative comment on {l}"
#             neu = f"This is a neutral comment on {l}"

#         # Determine sentiment using zero-shot classification
#         res2 = nli(s, candidate_labels=[pos, neg, neu])
#         polarity = res2['labels'][0]
#         results.append({
#             "sentence": s,
#             "cat": l,
#             "polarity": polarity[10:13],  # Extract polarity label
#             "polarity_score": res2['scores'][0]
#         })

# print(f"Processed {scount} sentences.")

# df_results = pd.DataFrame(results)
# summary = pd.crosstab(df_results['cat'], df_results['polarity'])

# df_summary = pd.DataFrame(summary)
# df_summary["category"] = df_summary.index

# df_summary["pos"] = df_summary["pos"] if "pos" in df_summary.columns else 0
# df_summary["neg"] = df_summary["neg"] if "neg" in df_summary.columns else 0
# df_summary["neu"] = df_summary["neu"] if "neu" in df_summary.columns else 0

# # Calculate the total
# df_summary["total"] = df_summary["pos"]+df_summary["neg"]+df_summary["neu"]
# # represent overall sentiment for the categary based on num of pos/neg
# df_summary["sentiment"] = np.round((df_summary["pos"]-df_summary["neg"])/(df_summary["neg"]+df_summary["pos"]),2)

# df_results
# df_summary