import pandas as pd
# import model from hugging face

# hardcoded variables
## ...

def aspectClassification(self, rawInput_file):

    df = rawInput_file
    # ... pre-processing steps to strip each review into multiple sentences
    # ... run classification model for each sentence

    # TODO transfrom df to aspectOutput_df and replace hard-coding below
    
    data = {
        'Reviews': [
            'This printer sucks. I want a refund.',
            'This printer sucks. I want a refund.'
        ],
        'Sentence': [
            'This printer sucks.',
            'I want a refund.'
        ],
        'Aspect': [
            'Quality',
            'Cost'
        ]
    }

    aspectOutput_df = pd.DataFrame(data)

    return aspectOutput_df