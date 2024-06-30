import streamlit as st
from docs.resume import display_resume_pdf
from docs.biography import intro
from docs.projects import projects
from interactive.interactive_models import i_models
from interactive.pretrained_models import pretrained_models


def intro_page():

    st.write("# Hi All! Welcome to My Project, Classification of Dwarf and Giant Stars! 🌌🌟🪐")
    st.sidebar.success("Please select a Page View.")

    st.markdown(
        """
        This project attempts to perform a Classification of Giant stars, Dwarf Stars, and group
        other stars into an 'Other' Category.

        You will see the impacts of broadly classifying categories vs narrowing the scope of classification
        and the impact on performance of the model. The following Classification models will 
        be tested:
                                1. Decision Trees
                                2. Random Forest
                                3. K Nearest Neighbors

        Giant and Dwarfs are the few of various star categories that describe the evolutionary status 
        of a star. These stars have distinct characteristics and differ greatly from one another. 
        These are stars that belong on the Main Sequence of stars, which are stars that fuse hydrogen to helium 
        at its core. Main Sequence Stars are predictable in their evolutionary status - stars 
        above a certain threshold will grow into Giant stars, while others below the threshold will 
        become dwarfs. 

        Below is a diagram of an H-R Diagram. This describes main sequence stars and their
        evolutionary status.
	
    """
    )

    st.image('app/docs/hrdiagram1.jpg', caption='Hertzsprung-Russell Diagram', use_column_width=True)

    st.markdown(
        """
        The scope of the project is to try to properly classify the dataset into the categories of 
        1. Giant Stars 2. Dwarf stars and 3. Other stars. Doing so will validate that based on the 
        metrics of brightness (Absolute Magnitude) and Color (B-V Index) we can systematically 
        categorize these stars into their respective evolutionary status. This will also allow 
        us to ultimately determine the death of these stars. 


          The fundamental goals of this project is to the final the optimal model with an accuracy of over 80%. 
          The impacts of the models performance will be discussed and the optimal model will be 
          chosen.
    """
    )

def resume():
    """Function to call display_resume_pdf(file)"""
    return display_resume_pdf('app/docs/Resume.pdf')

pages = {
    "Introduction": intro_page,
    "Biography": intro,
    'Resume': resume,
    'Projects': projects,
    'Pre-trained models': pretrained_models,
    'Interactive Models': i_models
}

page_title = st.sidebar.selectbox("Choose a page", pages.keys(), index=0)
pages[page_title]()