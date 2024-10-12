from controller import controllerService

# hard-coded variables
max_files = 100 # this refers to the max log files recoded in the repository.
rawInput_filepath = "./dataSource/combined_df.csv"
aspectOutput_filepath = "./models/aspectClassification/output_df.csv"
sentimentOutput_filepath = "./models/sentimentExtraction/output_df2.csv"
overallResultsOutput_filepath = "./results/output_df3.csv"

## specifically for Streamlit
# binSize = 10 # this setting is to adjust the bin size of Adjusted Score's histogram chart
# section = 21 # this refers to the no. of SSIC codes in this hierarchy (from DoS).
# division = 81 # this refers to the no. of SSIC codes in this hierarchy (from DoS).
# group = 204 # this refers to the no. of SSIC codes in this hierarchy (from DoS).
# Class = 382 # this refers to the no. of SSIC codes in this hierarchy (from DoS).
# subclass = 1032 # this refers to the no. of SSIC codes in this hierarchy (from DoS).

modelResults = controllerService(maxFiles = max_files)

logger = modelResults.setup_logger()
logger.info('Start code execution ...')

modelResults.runAspectClassification(logger, rawInput_filepath, aspectOutput_filepath)
modelResults.runSentimentExtraction(logger, aspectOutput_filepath, sentimentOutput_filepath, overallResultsOutput_filepath)