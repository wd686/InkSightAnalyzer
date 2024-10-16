from models.aspectClassification import aspectClassification
from models.sentimentExtraction import sentimentExtraction

class controllerService:

    def __init__(self):
        return
    
    def runAspectClassification(self, rawInput_file):
        return aspectClassification(self, rawInput_file)

    def runSentimentExtraction(self, aspectInput_df):
        return sentimentExtraction(self, aspectInput_df)