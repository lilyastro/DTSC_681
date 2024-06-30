from sklearn.neighbors import KNeighborsClassifier

class KNeighbors:
    def __init__(self, n_neighbors=5, metric='euclidean', weights=None):
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors, metric = metric, weights=weights)

    def train(self, X_train, y_train):
        # Train the k-NN model
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        # Make predictions on the test data
        return self.model.predict(X_test)