import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# hard-coded variables
# df = pd.read_excel('dataSource/Combined Survey+Web review.xlsx')
startEndPeriods_df = pd.read_csv('Sandbox/streamlitProcessing/generatedCSVs/startEndPeriods.csv')
reviewSource_df = pd.read_csv('Sandbox/streamlitProcessing/generatedCSVs/reviewSource.csv')
inkSupply_df = pd.read_csv('Sandbox/streamlitProcessing/generatedCSVs/inkSupply.csv')
printer_df = pd.read_csv('Sandbox/streamlitProcessing/generatedCSVs/printer.csv')
supplies_df = pd.read_csv('Sandbox/streamlitProcessing/generatedCSVs/supplies.csv')
ageGender_df = pd.read_csv('Sandbox/streamlitProcessing/generatedCSVs/ageGender.csv')
sentimentTime_df = pd.read_csv('Sandbox/streamlitProcessing/generatedCSVs/sentimentTime.csv')

# Set page config
st.set_page_config(
    page_title='Past Trends', # Set display name of browser tab
    layout="wide", # "wide" or "centered"
    initial_sidebar_state="expanded"
)

# df2 = df.drop_duplicates(subset = 'Survey ID')[['LTR', 'Star Rating', 'Review Source', 'Supplies Family', 'Printer Family', 'Combined Text', 'Ink Supply Type', 'Month of Response Date', 'Age Range', 'Gender']]
# df2['Month of Response Date'] = pd.to_datetime(df2['Month of Response Date'])

# # filter off validation data
# df2 = df2[df2['Month of Response Date'] < '2024-04-01']

# reviewSource pie chart df
# df2.loc[(df2['Review Source'].notnull()) & (df2['Review Source'].str.contains('amazon', case = False)), 'Review Source'] = 'Amazon'
# reviewSource_df = df2.groupby('Review Source').count().reset_index()

# inkSupply pie chart df
# inkSupply_df = df2.groupby('Ink Supply Type').count().reset_index()

# printer family bar chart df
# df2['Printer Family'] = df2['Printer Family'].str.strip().str.title()
# printer_df = df2.groupby('Printer Family').count().sort_values(ascending = False, by = 'Printer Family').reset_index()

# supplies family bar chart df
# df2['Supplies Family'] = df2['Supplies Family'].str.strip().str.title()
# supplies_df = df2.groupby('Supplies Family').count().sort_values(ascending = False, by = 'Supplies Family').reset_index()

# age/ gender stacked bar chart df
# ageGender_df = df2[(df2['Age Range'].notnull()) & (df2['Gender'].notnull())][['Age Range', 'Gender']].reset_index(drop = True)
# ageGender_df = ageGender_df[((ageGender_df.Gender == 'Male') | (ageGender_df.Gender == 'Female')) & (~(ageGender_df['Age Range'] == 'Prefer not to answer'))]

# sentiment/ time stacked bar chart df
# def score_to_sentiment(row):
#     if not pd.isna(row['LTR']):
#         # Use LTR (0-10)
#         if row['LTR'] <= 6:
#             return 'Negative'
#         else:
#             return 'Positive'
#     elif not pd.isna(row['Star Rating']):
#         # Use Star Rating (1-5)
#         if row['Star Rating'] <= 3:
#             return 'Negative'
#         else:
#             return 'Positive'
#     else:
#         return 'Unknown'
# sentiment_list = []
# for index, row in df2.iterrows():
#     sentiment_list.append(score_to_sentiment(row))
# df2['sentiment'] = sentiment_list
# sentimentTime_df = df2[df2.sentiment.isin(['Negative', 'Positive'])][['sentiment', 'Month of Response Date']]

# # extract start-end periods
# startEndPeriods_df = pd.concat([df2.sort_values(by = 'Month of Response Date')['Month of Response Date'].dt.strftime('%B %Y').head(1),
#                                 df2.sort_values(by = 'Month of Response Date')['Month of Response Date'].dt.strftime('%B %Y').tail(1)], axis =0)
startPeriod_str = startEndPeriods_df.head(1).reset_index()['Month of Response Date'][0]
endPeriod_str = startEndPeriods_df.tail(1).reset_index()['Month of Response Date'][0]

st.header('Visualization of Training Data')
st.write(f"Time period: *{startPeriod_str} - {endPeriod_str}*")

col1, col2 = st.columns([1,1])

with col1:

    # Extract labels and values from the DataFrame
    labels = reviewSource_df['Review Source']
    sizes = reviewSource_df['LTR']

    # Calculate the total n value
    total_value = sum(sizes)

    # Create the pie chart
    fig, ax = plt.subplots(figsize=(5, 5))  # Adjust figure size
    ax.pie(
        sizes, 
        labels=labels, 
        autopct='%1.1f%%', 
        startangle=90, 
        wedgeprops=dict(edgecolor='black', linewidth=2.5)  # Outline pie slices with black border
    )

    # Equal aspect ratio ensures that the pie is drawn as a circle
    ax.axis('equal')

    # Add a title with the total n value, bold font, and higher position
    plt.title(f"Review Source Distribution", fontsize=14, fontweight='bold', pad=30)

    # Display the pie chart in Streamlit
    st.pyplot(fig)


    # Create horizontal bar chart
    fig, ax = plt.subplots(figsize=(20, 15))  # Set the figure size
    bars = ax.barh(printer_df['Printer Family'], printer_df['LTR'], color='skyblue')  # Create horizontal bars

    # Add title
    ax.set_title('Printer Family Distribution', fontweight='bold', pad=10)  # Title of the chart

    # Remove right and top spines
    ax.spines[['right', 'top']].set_visible(False)  # Remove spines

    # Add data values on top of each bar
    for bar in bars:
        ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2,  # Position the text
                f'{bar.get_width():.0f}',  # The value to display
                va='center',  # Center align vertically
                ha='left')  # Align the text to the left

    # Adjust y-axis limits to remove gaps
    ax.set_ylim(-0.5, len(printer_df) - 0.2)  # Set limits to fit bars tightly

    # Display the chart in Streamlit
    st.pyplot(fig)

with col2:

    # Extract labels and values from the DataFrame
    labels = inkSupply_df['Ink Supply Type']
    sizes = inkSupply_df['LTR']

    # Calculate the total n value
    total_value = sum(sizes)

    # Create the pie chart
    fig, ax = plt.subplots(figsize=(5, 5))  # Adjust figure size
    ax.pie(
        sizes, 
        labels=labels, 
        autopct='%1.1f%%', 
        startangle=90, 
        wedgeprops=dict(edgecolor='black', linewidth=2.5)  # Outline pie slices with black border
    )

    # Equal aspect ratio ensures that the pie is drawn as a circle
    ax.axis('equal')

    # Add a title with the total n value, bold font, and higher position
    plt.title(f"Ink Supply Distribution", fontsize=14, fontweight='bold', pad=30)

    # Display the pie chart in Streamlit
    st.pyplot(fig)


    # Create horizontal bar chart
    fig, ax = plt.subplots(figsize=(20, 15))  # Set the figure size
    bars = ax.barh(supplies_df['Supplies Family'], supplies_df['LTR'], color='skyblue')  # Create horizontal bars

    # Add title
    ax.set_title('Supplies Family Distribution', fontweight='bold', pad=10)  # Title of the chart

    # Remove right and top spines
    ax.spines[['right', 'top']].set_visible(False)  # Remove spines

    # Add data values on top of each bar
    for bar in bars:
        ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2,  # Position the text
                f'{bar.get_width():.0f}',  # The value to display
                va='center',  # Center align vertically
                ha='left')  # Align the text to the left

    # Adjust y-axis limits to remove gaps
    ax.set_ylim(-0.5, len(supplies_df) - 0.2)  # Set limits to fit bars tightly

    # Display the chart in Streamlit
    st.pyplot(fig)


# Define the custom order for age ranges
age_order = ['Under 18', '18-24 years', '25-34 years', '35-50 years', '51-65 years', 'Over 65 years']

# Count occurrences of each Gender for each Age Range
age_gender_counts = ageGender_df.groupby(['Age Range', 'Gender']).size().unstack(fill_value=0)

# Reindex to ensure all categories are included
age_gender_counts = age_gender_counts.reindex(age_order, fill_value=0)

# Convert 'Age Range' to categorical with specified order
age_gender_counts.index = pd.CategoricalIndex(age_gender_counts.index, categories=age_order, ordered=True)

# Create the stacked bar chart
fig, ax = plt.subplots(figsize=(13, 7))
colors = ['#FF6F91', '#8470FF']
age_gender_counts.plot(kind='bar', stacked=True, ax=ax, color=colors)

# Remove right and top spines
ax.spines[['right', 'top']].set_visible(False)  # Remove spines

# Add data labels in each bar segment
for p in ax.patches:
    height = p.get_height()
    width = p.get_width()
    x = p.get_x()
    y = p.get_y() + height / 2  # Center label vertically within segment
    ax.text(x + width / 2, y, f"{height:.0f}", ha='center', va='center', color='black', fontsize=10)

# Add labels and title
ax.set_title('Distribution of Gender by Age Range', fontweight='bold', fontsize=17, pad=10)  # Increase font size for title
ax.set_xticklabels(age_order, rotation=0)  # Rotate x labels for better readability
ax.legend(title='Gender', fontsize=12, title_fontsize='14')  # Increase font size for legend and title

# Adjust layout for a tight fit
plt.tight_layout()

# Display the plot in Streamlit
st.pyplot(fig)


# Count occurrences
time_Sentiment_counts = sentimentTime_df.groupby(['Month of Response Date', 'sentiment']).size().unstack(fill_value=0)

# Format month index
time_Sentiment_counts.index = pd.to_datetime(time_Sentiment_counts.index).strftime("%b '%y")

# Create the stacked bar chart
fig, ax = plt.subplots(figsize=(13, 7))
colors = ['#B22222', '#3CB371']  # red and green
time_Sentiment_counts.plot(kind='bar', stacked=True, ax=ax, color=colors)

# Remove right and top spines
ax.spines[['right', 'top']].set_visible(False)

# Add data labels in each bar segment
for p in ax.patches:
    height = p.get_height()
    width = p.get_width()
    x = p.get_x()
    y = p.get_y() + height / 2  # Center label vertically within segment
    ax.text(x + width / 2, y, f"{height:.0f}", ha='center', va='center', color='black', fontsize=10)

# Add labels and title
ax.set_title('Distribution of Sentiments over Time', fontweight='bold', fontsize=17, pad=10)  # Increase font size for title
ax.set_xticklabels(time_Sentiment_counts.index, rotation=0)  # Rotate x labels for better readability
ax.legend(title='Sentiment', fontsize=12, title_fontsize='14')  # Increase font size for legend and title

# Adjust layout for a tight fit
plt.tight_layout()

# Display the plot in Streamlit
st.pyplot(fig)