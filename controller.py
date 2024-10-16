from models.aspectClassification.aspectClassification import aspectClassification
from models.sentimentExtraction.sentimentExtraction import sentimentExtraction

class controllerService:

    def __init__(self):
        return
    
    def runAspectClassification(self, rawInput_file):
        aspectClassification(self, rawInput_file)

    def runSentimentExtraction(self, aspectInput_df):
        sentimentExtraction(self, aspectInput_df)