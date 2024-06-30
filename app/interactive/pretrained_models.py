import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from model.prep.clean_data import CleanData
from model.prep.prep_data import PrepData
from model.prep.performance_data import PerformanceData
from model.decision_tree import DecisionTree
from model.random_forest import RandomForest
from model.k_nearest_neighbors import KNeighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix


def pretrained_models():
    # Clean the Data using CleanData Class
    clean_data = CleanData('app/model/prep/Star99999_raw.csv')
    df = clean_data.filter_df(['Vmag', 'Plx', 'e_Plx', 'B-V', 'SpType'])

    st.header('Data Overview Prior Cleaning')
    st.markdown('**Preview of Data Prior to Cleaning...**')
    st.dataframe(df.head(10), use_container_width=True)

    st.markdown('**Stats of Data Prior to Cleaning...**')
    st.dataframe(df.describe(), use_container_width=True)
    
    st.header('Clean Dataset')
    st.markdown(
        """
        **We will perform the following steps:**     
        1. Convert datatypes from string to floats for numerical values  
        2. Drop any null values - can see there are nulls in the column SpType. We will drop these,  
          this is a small percentage of the sample (97k not null / 99k.)  
        3. Drop any outlier errors of parallax. This will be defined as 3 standard deviations away from the mean  
         of error parallax.  
        4. Add our Absolute magnitude Column to help classify the classes of Dwarfs/Giants/Others. This is a measurement  
        of brightness.  
                    Absolute Magnitude = Vmag + 5(log10plx+1)  
          
        """
    )

    df = clean_data.convert_datatypes()
    df = clean_data.drop_nulls()
    df = clean_data.drop_outlier_plx()
    df = clean_data.add_abs_mag()
    #reset the index now that the dataframe has been altered
    df.reset_index(inplace=True, drop=True)

    st.title('White and Giant Dwarf Classification')

    st.header('Data Overview After Cleaning')
    st.write('Preview of Data Prior to Cleaning...')
    st.dataframe(df.head(10), use_container_width=True)

    st.write('Stats of Data After Cleaning...')
    st.dataframe(df.describe(), use_container_width=True)

    # Prep the Data using PrepData Class
    prep_data = PrepData(df)
    prep_data.classify()

    #visualize the distribution of the classes
    st.header('Classes from Dataset')
    st.write("""**Visualizing the distribution of the Target Class:**
             0: Dwarf
             1: Giant
             3: Other
    """)
    fig, ax = plt.subplots()
    sns.histplot(df['Target'], kde=False, ax=ax)
    ax.set_title('Target Distribution')
    ax.set_xlabel('Target')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)

    st.markdown(""":red[Significantly more Other and Giants. We need to balance dataset prior
            to classification.]""")
    
    #apply balancing to the dataset since there are significantly more stars 
    #that fall under 'Other' or 'Giant'
    df = prep_data.balance()

    st.header('After Balancing ⚖️')
    fig, ax = plt.subplots()
    sns.histplot(df['Target'], kde=False, ax=ax)
    ax.set_title('Target Distribution')
    ax.set_xlabel('Target')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)

    #split the data into test and training sets
    X, Y = prep_data.split(df, ['Vmag', 'Plx', 'B-V', 'e_Plx', 'B-V', 'Abs_Mag'], ['Target'])
    X_encoded = prep_data.encode(X, 4)
    #train both types - scaled and non scaled. Scaled will be for K Neighbors
    X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = prep_data.train(X_encoded, Y, True)
    X_train, X_test, y_train, y_test = prep_data.train(X_encoded, Y, False)

    #initialize benchmark
    benchmark = PerformanceData()

    #define a function that can be reused to run all the classification models.
    def run_all_classifications(rf_params, dt_params, kn_params, binary=False):
        models = {
            'Random Forest': RandomForest(**rf_params),
            'Decision Tree': DecisionTree(**dt_params),
            'K Nearest Neighbors': KNeighbors(**kn_params)
        }
        results = {}
        for name, model in models.items():
            if name == 'K Nearest Neighbors': # sensitive to scaling.
                model.train(X_train_scaled, y_train_scaled)
                y_preds = model.predict(X_test_scaled)
                if binary:
                    metrics = benchmark.get_performance_metrics(name, y_test_scaled, y_preds, True) # 2 classes
                else:
                    metrics = benchmark.get_performance_metrics(name, y_test_scaled, y_preds, False) # 3 classes
            else:
                model.train(X_train, y_train)
                y_preds = model.predict(X_test)
                if binary:
                    metrics = benchmark.get_performance_metrics(name, y_test, y_preds, True) # 2 classes
                else:
                    metrics = benchmark.get_performance_metrics(name, y_test, y_preds, False) # 3 classes
            results[name] = metrics
            
            #plot a confusion matrix to show the classes
            cm = confusion_matrix(y_test, y_preds)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title(f'{name} Confusion Matrix')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            st.pyplot(fig)

        return results

    @st.cache_data
    #define a function to perform a grid search.
    # The ranges of parameters have been chosen.
    # define one function for first grid search, and one for second to cache the 
    # results of both for better performance.
    def perform_grid_search():
        dt_params = benchmark.grid_search(DecisionTreeClassifier(),
            {'max_depth': list(range(1, 15, 1)),
            'min_samples_split': list(range(2, 20, 2)),
            'splitter': ["best", "random"]}, X_train, y_train)
        
        rf_params = benchmark.grid_search(RandomForestClassifier(),
            {'n_estimators': list(range(1, 100, 50)),
            'max_depth': list(range(1, 10, 1)),
            'random_state': list(range(0, 100, 25))}, X_train, y_train)
        
        knn_params = benchmark.grid_search(KNeighborsClassifier(),
            {'n_neighbors': list(range(1, 15, 1)),
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']}, X_train_scaled, y_train_scaled)
        
        return dt_params, rf_params, knn_params
    
    @st.cache_data
    def perform_binary_grid_search():
        dt_params = benchmark.grid_search(DecisionTreeClassifier(),
            {'max_depth': list(range(1, 15, 1)),
            'min_samples_split': list(range(2, 20, 2)),
            'splitter': ["best", "random"]}, X_train, y_train)
        
        rf_params = benchmark.grid_search(RandomForestClassifier(),
            {'n_estimators': list(range(1, 100, 50)),
            'max_depth': list(range(1, 10, 1)),
            'random_state': list(range(0, 100, 25))}, X_train, y_train)
        
        knn_params = benchmark.grid_search(KNeighborsClassifier(),
            {'n_neighbors': list(range(1, 15, 1)),
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']}, X_train_scaled, y_train_scaled)
        
        return dt_params, rf_params, knn_params
    
    def plot_metrics(metrics_list, metric_name):
            model_names = ['Random Forest', 'Decision Tree', 'K Nearest Neighbors']
            if metric_name != 'F1 Score': # do not plot, it is an array
                values = [metrics[metric_name] for metrics in metrics_list.values()]
                fig, ax = plt.subplots()
                ax.bar(model_names, values)
                ax.set_title(f'{metric_name} for Models')
                ax.set_xlabel('Model')
                ax.set_ylabel(metric_name)
                st.pyplot(fig)
    
    #run initial classification
    st.header('Initial Classification')
    if st.button('Run All Classifications with initial Hyper parameters'):
        with st.spinner("Running Classification..."):
            results = run_all_classifications({"max_depth": 5, "n_estimators": 100, "random_state": 50}, {"max_depth": 5, "min_samples_split": 2, "splitter": 'best'}, {"n_neighbors": 5, "metric": 'euclidean', "weights": 'uniform'}, binary=False)
            st.markdown("""
                        Random Forest Hyperparameters initially chosen: max_depth = 5, n_estimators=100, random_state=50    
                        Decision Tree Hyperparameters initially chosen: max_depth = 5, min_samples_split=2, splitter=best    
                        K Nearest Neighbors Hyperparameters initially chosen:n_neighbors=5, metric='euclidean', weights='uniform'  
                        """)
            st.write('Performance metrics for all models:')
            st.write(results)
            st.header('Model Performance Comparison')

            plot_metrics(results, 'Accuracy')
            plot_metrics(results, 'Precision')
            plot_metrics(results, 'Recall')
            plot_metrics(results, 'F1 Score')

    # run grid search
    st.header('Run a Grid Search to find Optimal Hyper Parameters')
    st.markdown("""
                 Parameter Ranges:
                    Random Forest: {'n_estimators': list(range(1, 100, 25)), 'max_depth': list(range(1, 15, 1)), 'random_state': list(range(0, 100, 25))}  
                    Decision Tree: {'max_depth': list(range(1, 15, 1)), 'min_samples_split': list(range(2, 20, 2)), 'splitter': ["best", "random"]} 
                    K Nearest Neighbors: {'n_neighbors': list(range(1, 15, 1)), 'weights': ['uniform', 'distance'], 'metric': ['euclidean', 'manhattan']}  
    """)
    if st.button('Run a Grid Search for all Classification Models'):
        with st.spinner("Running Grid Search..."):
            dt_params, rf_params, knn_params = perform_grid_search()
            st.write('Decision Tree best parameters:')
            st.write(dt_params)
            st.write('Random Forest best parameters:')
            st.write(rf_params)
            st.write('K Nearest Neighbors best parameters:')
            st.write(knn_params)
            st.header('Classification')
    if st.button('Run All Classifications with optimal Hyper parameters found', type="secondary"):
        with st.spinner("Running Classification..."):
            dt_params, rf_params, knn_params = perform_grid_search()
            results = run_all_classifications(rf_params, dt_params, knn_params, binary=False)
            st.write('Performance metrics for all models:')
            st.write(results)
            plot_metrics(results, 'Accuracy')
            plot_metrics(results, 'Precision')
            plot_metrics(results, 'Recall')


            st.markdown(""":red[We did not meet our Accuracy goal of 80%. Take out 'other' category and just classify  
            Giant Stars and Dwarf Stars]  """)
            
    # Take away other category for classification, do binary for only Dwarf and Giants
    df = df[(df['Target'] == 1) | (df['Target'] == 0)]


    #split into test and training sets
    X, Y = prep_data.split(df, ['Vmag', 'Plx', 'B-V', 'e_Plx', 'B-V', 'Abs_Mag'], ['Target'])
    X_encoded = prep_data.encode(X, 4)
    #train both types - scaled and non scaled. Scaled will be for K Neighbors
    X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = prep_data.train(X_encoded, Y, True)
    X_train, X_test, y_train, y_test = prep_data.train(X_encoded, Y, False)
        
    st.header('Initial Binary Classification')
    if st.button('Run all binary classifications with initial hyper parameters'):
        with st.spinner("Running Classification..."):
            results = run_all_classifications({"max_depth": 5, "n_estimators": 100, "random_state": 50}, {"max_depth": 5, "min_samples_split": 2, "splitter": 'best'}, {"n_neighbors": 5, "metric": 'euclidean', "weights": 'uniform'}, binary=True)
            st.markdown("""
                        Random Forest Hyperparameters initially chosen: max_depth = 5, n_estimators=100, random_state=50    
                        Decision Tree Hyperparameters initially chosen: max_depth = 5, min_samples_split=2, splitter=best    
                        K Nearest Neighbors Hyperparameters initially chosen:n_neighbors=5, metric='euclidean', weights='uniform'  
                        """)
            st.write('Performance metrics for all models:')
            st.write(results)
            plot_metrics(results, 'Accuracy')
            plot_metrics(results, 'Precision')
            plot_metrics(results, 'Recall')


    st.header('Run a Grid Search to find Optimal Hyper Parameters for Binary Classification')
    st.markdown("""
                 Parameter Ranges:
                    Random Forest: {'n_estimators': list(range(1, 100, 25)), 'max_depth': list(range(1, 15, 1)), 'random_state': list(range(0, 100, 25))}  
                    Decision Tree: {'max_depth': list(range(1, 15, 1)), 'min_samples_split': list(range(2, 20, 2)), 'splitter': ["best", "random"]} 
                    K Nearest Neighbors: {'n_neighbors': list(range(1, 15, 1)), 'weights': ['uniform', 'distance'], 'metric': ['euclidean', 'manhattan']}  
    """)
    if st.button('Run a Grid Search for all Binary Classification Models'):
        with st.spinner("Running Grid Search..."):
            dt_params, rf_params, knn_params = perform_binary_grid_search()
            st.write('Decision Tree best parameters:')
            st.write(dt_params)
            st.write('Random Forest best parameters:')
            st.write(rf_params)
            st.write('K Nearest Neighbors best parameters:')
            st.write(knn_params)

    st.header('Binary Classification')
    if st.button('Run All Binary Classifications with optimal Hyper parameters found'):
        dt_params, rf_params, knn_params = perform_binary_grid_search()
        with st.spinner("Running Classification..."):
            results = run_all_classifications(rf_params, dt_params, knn_params, binary=True)
            st.write('Performance metrics for all models:')
            st.write(results)
            plot_metrics(results, 'Accuracy')
            plot_metrics(results, 'Precision')
            plot_metrics(results, 'Recall')
            plot_metrics(results, 'F1 Score')

            st.markdown(""":green[Hooray! We achieved our goal of over 80% accuracy on a
                        classification model with the most optimal being Random Forest! ]  """)
