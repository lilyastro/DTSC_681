import streamlit as st
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Sidebar for hyperparameter inputs
st.sidebar.header('Decision Tree Hyperparameters')
max_depth = st.sidebar.slider('Max Depth', min_value=1, max_value=10, value=2)
min_samples_split = st.sidebar.slider('Min Samples Split', min_value=2, max_value=10, value=2)

# Decision Tree model training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split)
clf.fit(X_train, y_train)

# Function to display Decision Tree Classifier page
def decision_tree_page():
    st.title('Decision Tree Classifier')
    st.write('### Hyperparameters')
    st.write(f'- Max Depth: {max_depth}')
    st.write(f'- Min Samples Split: {min_samples_split}')

    # Prediction
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f'### Model Accuracy: {accuracy:.2f}')

# Function to display Resume page
def resume_page():
    st.title('My Resume')
    # Add your resume content here

# Function to display Preset Model page
def preset_model_page():
    preset_inputs = {
        'max_depth': 3,
        'min_samples_split': 2
    }
    preset_clf = DecisionTreeClassifier(max_depth=preset_inputs['max_depth'], min_samples_split=preset_inputs['min_samples_split'])
    preset_clf.fit(X_train, y_train)
    preset_y_pred = preset_clf.predict(X_test)
    preset_accuracy = accuracy_score(y_test, preset_y_pred)

    st.title('Preset Model Performance')
    st.write('### Hyperparameters')
    st.write(f'- Max Depth: {preset_inputs["max_depth"]}')
    st.write(f'- Min Samples Split: {preset_inputs["min_samples_split"]}')
    st.write(f'### Model Accuracy: {preset_accuracy:.2f}')

# Main function to run the Streamlit app
def main():
    st.sidebar.title('Navigation')
    page = st.sidebar.selectbox('Go to', ['Decision Tree', 'Resume', 'Preset Model'])

    if page == 'Decision Tree':
        decision_tree_page()
    elif page == 'Resume':
        resume_page()
    elif page == 'Preset Model':
        preset_model_page()

if __name__ == "__main__":
    main()
