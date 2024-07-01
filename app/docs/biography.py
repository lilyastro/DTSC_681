def intro():
    import streamlit as st

    st.title("Hi Everyone, I'm Lily! 👋")

    
    st.title("Introduction")

    st.markdown(
        """
        I was born in Cincinnati, Ohio. I lived there until 4th grade when I moved to the Columbus, OH area. I went to college 
        at Ohio State University and recieved my Bachelors of Science in 2021 in Astronomy and Astrophysics with a minor in
        Philosophy.
    """
    )

    st.image('app/docs/grad_pic.jpeg', caption='Graduation!', use_column_width=True)


    st.title("Professsional Background")

    st.markdown(
        """
        I currently work at a digital marketing agency as a Data Engineer. Upon my graduation, I am transitioning into a Machine
        Learning Engineering role on our AI/ML team. 

        In my free time, I like to go on hikes (although hikes in Columbus, OH are scarce), hang out with friends, and cook.
        I started doing Lagree (a form of Pilates) about 6 months ago and love it - I go about 3-4 times a week.
    """
    )

    st.title("Personal Life")

    st.markdown(
        """
        I currently reside in Columbus, OH with my Fiance. We are getting married next year in Tuscany, Italy
        and we can't wait! 
    """
    )

