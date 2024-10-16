import pandas as pd
# import model from hugging face

# hardcoded variables
## ...

def sentimentExtraction(self, aspectInput_df):

    df = aspectInput_df

    # ... run sentiment model for each sentence (containing 1 aspect) to obtain sentiment
    aspectSentimentOutput_df = df.copy()
     
    # ... aggregate results into final results output
    overallResultsOutput_df = aspectSentimentOutput_df.copy()
    
    return aspectSentimentOutput_df, overallResultsOutput_df # aspect-sentiment result outputs; aggregated final outputs