import pandas as pd
# import model from hugging face

# hardcoded variables
## ...

def aspectClassification(self, rawInput_file):

    print(type(rawInput_file))

    df = rawInput_file.copy()
    # ... pre-processing steps to strip each review into multiple sentences
    # ... run classification model for each sentence
    aspectOutput_df = df.copy()

    return aspectOutput_df