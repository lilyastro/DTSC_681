from sklearn.tree import DecisionTreeClassifier

class DecisionTree:
    def __init__(self, random_state=42, max_depth=None, min_samples_split=0.5, splitter='best'):
        self.model = DecisionTreeClassifier(random_state=random_state, max_depth = max_depth, min_samples_split=min_samples_split, splitter=splitter)

    def train(self, X_train, y_train):
        # Train the decision tree model
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        # Make predictions on the test data
        return self.model.predict(X_test)