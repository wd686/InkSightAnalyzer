import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# hard-coded variables
df = pd.read_excel('dataSource/Combined Survey+Web review.xlsx')

st.header('Visualization of Training Data')
st.write(f"*Period between December 2023 - March 2024*")

df2 = df[['LTR', 'Star Rating', 'Review Source', 'Supplies Family', 'Printer Family', 'Combined Text', 'Ink Supply Type', 'Month of Response Date', 'Age Range', 'Gender']].drop_duplicates(subset = 'Combined Text')
df2['Month of Response Date'] = pd.to_datetime(df2['Month of Response Date'])
df2 = df2[df2['Month of Response Date'] < '2024-04-01']

# reviewSource pie chart df
df2.loc[(df2['Review Source'].notnull()) & (df2['Review Source'].str.contains('amazon', case = False)), 'Review Source'] = 'Amazon'
reviewSource_df = df2.groupby('Review Source').count().reset_index()

col1, col2 = st.columns([1,1])

# inkSupply pie chart df
inkSupply_df = df2.groupby('Ink Supply Type').count().reset_index()

# supplies family bar chart df
df2['Supplies Family'] = df2['Supplies Family'].str.strip().str.title()
supplies_df = df2.groupby('Supplies Family').count().sort_values(ascending = False, by = 'Supplies Family').reset_index()

# printer family bar chart df
df2['Printer Family'] = df2['Printer Family'].str.strip().str.title()
printer_df = df2.groupby('Printer Family').count().sort_values(ascending = False, by = 'Printer Family').reset_index()

with col1:

    # Extract labels and values from the DataFrame
    labels = reviewSource_df['Review Source']
    sizes = reviewSource_df['LTR']

    # Calculate the total n value
    total_value = sum(sizes)

    # Create the pie chart
    fig, ax = plt.subplots(figsize=(6, 6))  # Adjust figure size
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
    fig, ax = plt.subplots(figsize=(16, 10))  # Set the figure size
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

with col2:

    # Extract labels and values from the DataFrame
    labels = inkSupply_df['Ink Supply Type']
    sizes = inkSupply_df['LTR']

    # Calculate the total n value
    total_value = sum(sizes)

    # Create the pie chart
    fig, ax = plt.subplots(figsize=(6, 6))  # Adjust figure size
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
    fig, ax = plt.subplots(figsize=(16, 10))  # Set the figure size
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