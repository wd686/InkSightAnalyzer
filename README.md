<img src="https://github.com/wd686/InkSightAnalyzer/blob/main/images/group_logo.jpg" width="170" height="170">

# Background
HP Inc, a global leader in printing solutions, is committed to deliver high quality products, services and customer experiences. One of the key components of HP’s printing trabusiness is its ink cartridges, an integral part to the performance of its printers. Ensuring customers’ satisfaction with the products is crucial to maintaining HP’s market leadership and brand loyalty.

To continuously improve its offerings, HP actively gathers customer feedback through multiple channels, including social media, online reviews, and customer surveys.  The feedback often includes valuable insights but also contains a mix of positive, negative and neutral sentiments. The key challenge is to systematically analyze and interpret the vast amount of data from the different review sources such as Amazon, Walmart and Office Depot. 

In this project, the team will be undertaking an in-depth analysis of customer sentiment related to HP’s printers and ink cartridges through Aspect Based Sentiment Analysis (ABSA) modelling.

# Directory
Please refer to this [link](https://github.com/wd686/InkSightAnalyzer/blob/main/Project%20Summary.pdf) for a high-level summary of the project.

The folders in this repository are described as follows:

- Data Sources (dataSource)
  - Data source used for the project.
- Images (images)
  - Contains the group logo.
- Deployed Models (models)
  - Models that are chosen for production (DistilBERT for multi-aspects classification & Facebook-BART NLI for sentiment analysis).
- Streamlit Pages (pages)
  - Codes for generating Streamlit webpage.
- Sandbox
  - Stash of all draft codes and output files (EDA/ Data Processing, model training, streamlit processing).

 The codes in this repository are used for deployment on [Streamlit](https://inksightanalyzer.streamlit.app/). Please refer to the webpage for more information and usage instructions.

# Contributors
1. Ang Mei Chi
2. Darius Chan Yak Weng
3. Lee Kuan Teng Roy
4. Liu Wudi
5. Michael Wong Wai Kit

# References
1. [Hugging Face fine-tuned models](https://huggingface.co/nusebacra)
2. [DistilBERT documentation](https://huggingface.co/docs/transformers/en/model_doc/distilbert)
3. [Facebook-BART NLI documentation](https://huggingface.co/facebook/bart-large-mnli)
4. [Libraries used](https://github.com/wd686/InkSightAnalyzer/blob/main/requirements.txt)
5. [Streamlit 'download CSV button' guide](https://www.youtube.com/watch?v=eJWHFJSjD9E)
6. [Streamlit 'upload CSV button' guide](https://www.youtube.com/watch?v=i3Ad3-Z-zbY)
