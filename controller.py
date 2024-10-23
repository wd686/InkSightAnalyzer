from models.aspectClassification import aspectClassification
from models.sentimentAnalyzer import sentimentAnalyzer

class controllerService:

    def __init__(self):
        return
    
    def runAspectClassification(self, rawInput_file):
        return aspectClassification(self, rawInput_file)

    def runsentimentAnalyzer(self, aspectSentimentOutput_df):
        return sentimentAnalyzer(self, aspectSentimentOutput_df)