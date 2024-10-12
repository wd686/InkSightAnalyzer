import pandas as pd
# import model from hugging face

# hardcoded variables
## ...

def sentimentExtraction(self, logger, aspectOutput_filepath, sentimentOutput_filepath, overallResultsOutput_filepath):

    combined_df = pd.read_csv(aspectOutput_filepath)

    # ... run sentiment model for each sentence (containing 1 aspect) to obtain sentiment
    # ... aggregate results into final results output

    combined_df.to_csv(sentimentOutput_filepath) # sentiment result outputs
    logger.info('Sentiment output file generated')
    combined_df.to_csv(overallResultsOutput_filepath) # aggregated final outputs
    logger.info('Final output file generated')