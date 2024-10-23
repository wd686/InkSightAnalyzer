<img src="https://github.com/wd686/InkSightAnalyzer/blob/main/images/group_logo.jpg" width="170" height="170">

# Background
HP Inc, a global leader in printing solutions, is committed to deliver high quality products, services and customer experiences. One of the key components of HP’s printing trabusiness is its ink cartridges, an integral part to the performance of its printers. Ensuring customers’ satisfaction with the products is crucial to maintaining HP’s market leadership and brand loyalty.

To continuously improve its offerings, HP actively gathers customer feedback through multiple channels, including social media, online reviews, and customer surveys.  The feedback often includes valuable insights but also contains a mix of positive, negative and neutral sentiments. The key challenge is to systematically analyze and interpret the vast amount of data from the different review sources such as Amazon, Walmart and Office Depot. 

In this project, the team will be undertaking an in-depth analysis of customer sentiment related to HP’s printers and ink cartridges through Aspect Based Sentiment Analysis (ABSA) modelling.

# Directory

Please refer to this [link](https://github.com/wd686/InkSightAnalyzer/blob/main/Project Summary.pdf) for a high-level summary of the project.

The folders in this repository are described as follows:

- Workflows (.github/workflows)
  - This folder include all .yml files which interacts with GitHub Actions to activate the respective workflows.
- Data Sources (dataSources)
  - Data sources are obtained from point 1 & 2 in the reference list below.
- Data Wrangling (dataWrangling)
  - Scripts in this folder carried out steps such as data cleaning (e.g. null values and non-English words removal), data prepatory work for text analytics, and exploratory data analysis.
- Data Modelling (models)
  - Analytical techniques explored includes Decision Tree, Random Forest, and Content-Based Recommender System.
- Images (images)
  - Contains Apple App Store, Google Play Store, and our group logos.
 
# Contributors
1. Ang Mei Chi
2. Lee Kuan Teng Roy
3. Liu Wudi
4. Michael Wong Wai Kit
5. Ong Wee Yang

# References
1. [Application ID list for Apple App Store](https://github.com/gauthamp10/apple-appstore-apps)
2. [Application ID list for Google Play Store](https://github.com/gauthamp10/Google_Play_App_Info)
4. [Web Scrapping Codes for Apple App Store Reviews](https://github.com/glennfang/apple-app-reviews-scraper/blob/main/src/apple_app_reviews_scraper.py)
5. [Google Play Store API guide](https://pypi.org/project/google-play-scraper)
8. [Running Python on GitHub Actions](https://www.python-engineer.com/posts/run-python-github-actions)


###
The codes in this repository are used for deployment on Streamlit (link). Please refer to the webpage for more information and usage instructions.
###
