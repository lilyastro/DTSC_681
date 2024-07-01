import streamlit as st
import base64


def display_resume():
    """Function to Display Resume (PDF)
    Args: None
    Returns: Obj: Streamlit Markdown display"""

    # did not work on streamlit cloud
    
    # # Open File
    # with open(file, "rb") as f:
    #     pdf = base64.b64encode(f.read()).decode('utf-8')

    # # Embedding PDF in HTML
    # pdf_fixed = F'<iframe src="data:application/pdf;base64,{pdf}" width="700" height="1000" type="application/pdf"></iframe>'
    # Displaying File
    return st.image('app/docs/Resume.jpg', caption='Resume', use_column_width=True)

