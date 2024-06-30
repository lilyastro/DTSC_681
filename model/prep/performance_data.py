from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
from sklearn.model_selection import GridSearchCV


class PerformanceData:
    """A Class that returns the grid search and accuracy for a respective model for White and Giant Dwarf classification.
    This is a prepartory """
    
    def __init__(self): 
        return
    
    def grid_search(self, model, param_grid, X_train, y_train, cv=5):
        """
        Perform grid search for hyperparameter tuning of a scikit-learn model DecisionTree, RandomForest, or K nearest Neighbors.

        Parameters:
        - model: Scikit-learn model.
        - param_grid: Dictionary of hyperparameters.
        - X_train: Training input samples.
        - y_train: Target values for X_train.
        - cv: Cross-validation strategy. Default is 5-fold cross-validation.

        Returns:
        - best_params: Best hyperparameters found during grid search.
        """
        # Initialize GridSearchCV with the specified model, parameter grid, and cross-validation.
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv, verbose=1, error_score='raise')
        
        # Fit GridSearchCV with the training data.
        grid_search.fit(X_train, y_train)
        
        # Print the best parameters and best score found by GridSearchCV.
        print(f"Best parameters found: {grid_search.best_params_}")
        
        # Return the best parameters found.
        return grid_search.best_params_
    
    def get_performance_metrics(self, model, y_test, y_pred):
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average=None) # for non binary classification
        prec = precision_score(y_test, y_pred, average='micro') # Calculate metrics globally 
        recall = recall_score(y_test, y_pred, average='micro') # Calculate metrics globally 
        performance_dict = {f' {model} F1 Score': f1,
            f' {model} Accuracy Score': acc,
            f' {model} Precision': prec,
            f' {model} Recal;': recall,
            }
            
        return performance_dict