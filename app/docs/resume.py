import streamlit as st
import base64


def display_resume_pdf(file):
    """Function to Display Resume (PDF)
    Args: None
    Returns: Obj: Streamlit Markdown display"""
    
    # Open File
    with open(file, "rb") as f:
        pdf = base64.b64encode(f.read()).decode('utf-8')

    #Embed
    html = (
        f'<iframe src="data:application/pdf;base64,{pdf}" '
        'width="800" height="1100" type="application/pdf"></iframe>'
    )
    # Displaying File
    return st.markdown(html, unsafe_allow_html=True)

