import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def interactive_models():

    # Load dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Sidebar - Classifier parameters
    st.sidebar.title("Classifier Parameters")
    classifier_choice = st.sidebar.selectbox("Choose Classifier", ["Decision Tree", "Random Forest", "K-Nearest Neighbors"])

    if classifier_choice == "Decision Tree":
        max_depth = st.sidebar.slider("Max Depth", min_value=1, max_value=10, value=3, step=1)
        criterion = st.sidebar.selectbox("Criterion", ["gini", "entropy"])
        clf = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion)

    elif classifier_choice == "Random Forest":
        n_estimators = st.sidebar.slider("Number of Estimators", min_value=1, max_value=100, value=10, step=1)
        max_depth = st.sidebar.slider("Max Depth", min_value=1, max_value=10, value=3, step=1)
        clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)

    elif classifier_choice == "K-Nearest Neighbors":
        n_neighbors = st.sidebar.slider("Number of Neighbors", min_value=1, max_value=15, value=5, step=1)
        clf = KNeighborsClassifier(n_neighbors=n_neighbors)

    # Main content
    st.title("Classifier Performance Evaluation")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Training and prediction
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Display performance
    st.write(f"### {classifier_choice} Performance")
    st.write(f"Accuracy: {accuracy:.2f}")

    # Display explanation if performance is bad
    if accuracy < 0.8:
        st.warning("The model performance is not satisfactory.")
        st.write("Possible reasons:")
        st.write("- Insufficient data")
        st.write("- Improper parameter tuning")
        st.write("- High dimensionality of data")
    else:
        st.success("The model performance is good!")

    # Show dataset information
    st.write("### Dataset Information")
    st.write(f"Number of samples: {X.shape[0]}")
    st.write(f"Number of features: {X.shape[1]}")
    st.write(f"Number of classes: {len(np.unique(y))}")
    st.write(f"Class labels: {np.unique(y)}")

    # Footer
    st.markdown("---")
    st.markdown("Created with ❤️ by [Your Name]")

