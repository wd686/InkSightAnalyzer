import pandas as pd
# import model from hugging face

# hardcoded variables
## ...

def aspectClassification(self, logger, rawInput_filepath, aspectOutput_filepath):

    combined_df = pd.read_csv(rawInput_filepath)
    logger.info('Read raw input file')

    # ... pre-processing steps to strip each review into multiple sentences
    # ... run classification model for each sentence

    combined_df.to_csv(aspectOutput_filepath) # input file for sentimentExtraction()
    logger.info('Aspect output file generated')
