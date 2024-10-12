import streamlit as st

# hard-coding
logoPath = 'images/group_logo.jpg'

# Set page config
st.set_page_config(
    page_title='InksightAnalyzer', # Set display name of browser tab
    # page_icon="🔍", # Set display icon of browser tab
    layout="wide", # "wide" or "centered"
    initial_sidebar_state="expanded"
)

# Define CSS styles
custom_styles = """
<style>
    .appview-container .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }


</style>
"""
# Display CSS styles using st.markdown
st.markdown(custom_styles, unsafe_allow_html=True)

# Visual Effects ### - https://docs.streamlit.io/develop/api-reference/status
# st.balloons() 

# st.sidebar.success("Explore our pages above ☝️")

# Display the logo image at the top left corner with a specific width
col1, col2, col3 = st.columns([1, 10, 1])  # Adjust column proportions as needed

with col1:
    st.image(logoPath, caption='', output_format='JPEG', width=150)  # Set width to shrink the image

st.write("## About this Webpage")

st.markdown(
    '''
This platform allows users to visualize the various aspects and its respetive sentiments obtained from a set of 'printer & ink' related customer surveys.\
 Additionally, users are also able to input adhoc text comments relating to 'printer & ink' and obtain its respective aspects and sentiments. 
'''
)

st.write("## Table of Contents")

st.markdown(
f'''
**Results**\n
This section allows users to visualize the various aspects and its respetive sentiments obtained from a set of 'printer & ink' related customer surveys.

**Prediction**\n
This section allows users to input adhoc text comments relating to 'printer & ink' and obtain its respective aspects and sentiments.

'''
)


