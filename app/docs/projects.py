import streamlit as st

def projects():

    st.title("Projects!")

    

    st.markdown(
    """
        Below is a link to my public project. This was my undergraduate thesis, where I idenfitied the state of Red Giant stars on their 
        evolutionary status- Red Giant or Red Clump. This work is in the process of being published, which I cannot share the paper yet.
    """
    )
    url = "https://kb.osu.edu/items/31a06acf-3b69-44ce-bd5d-2693fa311ca9"
    st.write("Senior Thesis: [link](%s)" % url)


    st.markdown(
    """
        Below is a link to my final github project. 
    """
    )
    url = "https://github.com/lilyastro/DTSC_681"
    st.write("Github Link: [link](%s)" % url)
