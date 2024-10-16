import pandas as pd
# import model from hugging face

# hardcoded variables
## ...

def sentimentExtraction(self, aspectInput_df):

    df = aspectInput_df

    # ... run sentiment model for each sentence (containing 1 aspect) to obtain sentiment
    # TODO transfrom df to aspectSentimentOutput_df and replace hard-coding below

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
        ],
        'Sentiment': [
            'Negative',
            'Negative'
        ]
    }

    aspectSentimentOutput_df = pd.DataFrame(data)
        
    # ... aggregate results into final results output
    # TODO transfrom aspectSentimentOutput_df to overallResultsOutput_df and replace hard-coding below

    data = {
        'Pos': [27, 3, 42, 53, 22],
        'Neg': [4, 0, 4, 7, 5],
        'Total': [31, 3, 46, 60, 27],
        'Category': ['Quality', 'Cost', 'Shipment', 'Services', 'Instant Ink'],
        'Sentiment': [0.05, 1.00, 0.83, 0.77, 0.00]
    }

    overallResultsOutput_df = pd.DataFrame(data)
    
    return aspectSentimentOutput_df, overallResultsOutput_df # aspect-sentiment result outputs; aggregated final outputs