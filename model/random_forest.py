from sklearn.ensemble import RandomForestClassifier

class RandomForest:
    def __init__(self, n_estimators=100, max_depth=None, random_state=None):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )

    def train(self, X_train, y_train):
        # Train the Random Forest model
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        # Make predictions on the test data
        return self.model.predict(X_test)