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
        'Pos': [4, 33, 42, 0, 22],
        'Neg': [54, 0, 3, 47, 22],
        'Total': [58, 33, 45, 47, 44],
        'Category': ['Quality', 'Cost', 'Shipment', 'Services', 'Instant Ink'],
        'Sentiment': [-0.86, 1.00, 0.87, -1.00, 0.00]
    }

    overallResultsOutput_df = pd.DataFrame(data)
    
    return aspectSentimentOutput_df, overallResultsOutput_df # aspect-sentiment result outputs; aggregated final outputs