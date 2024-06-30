import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from model.prep.clean_data import CleanData
from model.prep.prep_data import PrepData
from model.prep.performance_data import PerformanceData
from model.decision_tree import DecisionTree
from model.random_forest import RandomForest
from model.k_nearest_neighbors import KNeighbors
from sklearn.metrics import confusion_matrix

def i_models():

    st.title('White and Giant Dwarf Binary Classification')

    @st.cache_data
    def load_data():
        clean_data = CleanData('app/model/prep/Star99999_raw.csv')
        df = clean_data.filter_df(['Vmag', 'Plx', 'e_Plx', 'B-V', 'SpType'])
        df = clean_data.convert_datatypes()
        df = clean_data.drop_nulls()
        df = clean_data.drop_outlier_plx()
        df = clean_data.add_abs_mag()
        df.reset_index(inplace=True, drop=True)
        return df

    df = load_data()

    st.header('Dataframe Overview 🖼️')
    st.write('Displaying first few rows of the dataset after cleaning and preparation:')
    st.dataframe(df.head(), use_container_width=True)

    st.write('Displaying dataframe stats 📊')
    st.dataframe(df.head(), use_container_width=True)

    # Prep the Data
    prep_data = PrepData(df)
    prep_data.classify()
    df = prep_data.balance()
    # Take away other category for classification, do binary for only Dwarf and Giants
    df = df[(df['Target'] == 1) | (df['Target'] == 0)]


    st.header('Data Visualization 🌟')
    st.write('Visualizing the distribution of the target column balanced ⚖️')
    fig, ax = plt.subplots()
    sns.histplot(df['Target'], kde=False, ax=ax)
    ax.set_title('Target Column Distribution')
    ax.set_xlabel('Target')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)

    st.write('Splitting out into test and training set...')
    #split out into test and training set
    X, Y = prep_data.split(df, ['Vmag', 'Plx', 'B-V', 'e_Plx', 'B-V', 'Abs_Mag'], ['Target'])
    X_encoded = prep_data.encode(X, 4)
    X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = prep_data.train(X_encoded, Y, True)
    X_train, X_test, y_train, y_test = prep_data.train(X_encoded, Y, False)
    st.write('Data has been split into test and training set!...')


    benchmark = PerformanceData()

    #function to run classifcation for all models based on user input.
    def run_classification(model_name, params):
        if model_name == 'Random Forest':
            model = RandomForest(**params)
            model.train(X_train, y_train)
            preds = model.predict(X_test)
        elif model_name == 'Decision Tree':
            model = DecisionTree(**params)
            model.train(X_train, y_train)
            preds = model.predict(X_test)
        elif model_name == 'K Nearest Neighbors':
            model = KNeighbors(**params)
            model.train(X_train_scaled, y_train_scaled)
            preds = model.predict(X_test_scaled)
        
        metrics = benchmark.get_performance_metrics(model, y_test, preds)
        return metrics, preds

    st.header('Hyperparameters Selection')
    form = st.form("hyperparameters form")
    form.write("Select hyperparameters for each model:")
    # Random Forest Hyperparameters
    #define vast hyper parameter values to choose from.
    rf_max_depth = form.slider('Random Forest Max Depth', 1, 200, 100)
    rf_n_estimators = form.slider('Random Forest Number of Estimators', 1, 500, 250)
    rf_random_state = form.slider('Random Forest Random State', 0, 100, 50)

    # Decision Tree Hyperparameters
    dt_max_depth = form.slider('Decision Tree Max Depth', 1, 200, 25)
    dt_min_samples_split = form.slider('Decision Tree Min Samples Split', 1, 50, 25)
    dt_splitter = form.selectbox('Decision Tree Splitter', ['random', 'best'])

    # K Nearest Neighbors Hyperparameters
    knn_n_neighbors = form.slider('Number of Neighbors', 1, 50, 25)
    knn_metric = form.selectbox('Metric', ['manhattan', 'euclidean'])
    knn_weights = form.selectbox('Weights', ['distance', 'uniform'])

    submitted = form.form_submit_button("Run All Classifications")

    if submitted:
        with st.spinner("Running Classification..."):

            rf_params = {'max_depth': rf_max_depth, 'n_estimators': rf_n_estimators, 'random_state': rf_random_state}
            dt_params = {'max_depth': dt_max_depth, 'min_samples_split': dt_min_samples_split, 'splitter': dt_splitter}
            kn_params = {'n_neighbors': knn_n_neighbors, 'metric': knn_metric, 'weights': knn_weights}

            rf_metrics, rf_preds = run_classification('Random Forest', rf_params)
            dt_metrics, dt_preds = run_classification('Decision Tree', dt_params)
            kn_metrics, kn_preds = run_classification('K Nearest Neighbors', kn_params)

            st.write("### Random Forest Performance")
            st.write(rf_metrics)
            cm = confusion_matrix(y_test, rf_preds)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title('Random Forest Confusion Matrix')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            st.pyplot(fig)

            st.write("### Decision Tree Performance")
            st.write(dt_metrics)
            cm = confusion_matrix(y_test, dt_preds)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title('Decision Tree Confusion Matrix')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            st.pyplot(fig)

            st.write("### K Nearest Neighbors Performance")
            st.write(kn_metrics)
            cm = confusion_matrix(y_test, kn_preds)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title('K Nearest Neighbors Confusion Matrix')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            st.pyplot(fig)

            #function to plot the metrics for accuracy, precision, recall
            def plot_metrics(metrics_list, metric_name):
                model_names = ['Random Forest', 'Decision Tree', 'K Nearest Neighbors']
                if metric_name != 'F1 Score': # do not plot, it is an array
                    values = [metrics[metric_name] for metrics in metrics_list]
                    fig, ax = plt.subplots()
                    ax.bar(model_names, values)
                    ax.set_title(f'{metric_name} for Models')
                    ax.set_xlabel('Model')
                    ax.set_ylabel(metric_name)
                    st.pyplot(fig)

            #define a function to provide potential feedback if accuracy does not reach the goal of 80%
            def provide_feedback(model_name ,params, metrics):
                accuracy = metrics['Accuracy']
                if accuracy < 0.8:
                    st.markdown(f':red[Did not achieve 80% accuracy on our model: {model_name}. 😔 Here are a few suggestions...]')
                    if model_name == 'Random Forest':
                        if params['max_depth'] < 5:
                            st.markdown("**Inspect Max Depth hyperparameter. The max depth is too shallow, which might cause under-fitting for the Random Forest model . Try increasing.**")
                        if params['max_depth'] > 19: 
                            st.markdown("**Inspect Max Depth hyperparameter. The max depth is too high, which might cause over-fitting for the Random Forest model . Try decreasing.**")
                        if params['n_estimators'] < 50:
                            st.markdown("**Inspect Number of Estimators Hyperparameter. The number of estimators is too low. Try increasing.**")
                        if params['n_estimators'] > 200:
                            st.markdown("Inspect Number of Estimators Hyperparameter. The number of estimators is too high and too complex. Try decreasing.**")
                    if model_name == 'Decision Tree':
                        if params['max_depth'] < 5:
                            st.markdown("**Inspect Max Depth hyperparameter. The max depth is too shallow, which might cause under-fitting for the Decision Tree model. Try increasing.**")
                        if params['max_depth'] > 100:
                            st.markdown("**Inspect Max Depth hyperparameter. The max depth is too deep, which might cause over-fitting for the Decision Tree model. Try decreasing.**")
                        if params['min_samples_split'] > 20:
                            st.markdown("**Inspect Min Samples Split hyperparameter. The min samples split is too high, which might cause over-fitting. Try decreasing.**")
                        if params['min_samples_split'] < 3:
                            st.markdown("I**nspect Min Samples Split hyperparameter. The min samples split is too low, which might cause under-fitting. Try increasing.**")
                        if params['splitter'] == 'random':
                            st.markdown("**Inspect Splitter hyperparameter. The random splitter might not be finding the best way to split the tree. Try using 'best'.**")
                    if model_name == 'K Nearest Neighbors':
                        if params['n_neighbors'] < 5:
                            st.markdown("**Inspect Number of Neighbors hyperparameter. The number of neighbors is too low, which might cause a high variance. Try increasing.**")
                        if params['metric'] == 'euclidean':
                            st.markdown("**Inspect the Metric hyperparameter. The euclidean metric might not be capturing the best distances. Try using 'manhattan'.**")
                        if params['metric'] == 'manhattan':
                            st.markdown("**Inspect the Metric hyperparameter. The manhattan metric might not be capturing the best distances. Try using 'euclidean'.**")
                        if params['weights'] == 'uniform':
                            st.markdown("**Inspect the Weights hyperparameter. Uniform weights might not be optimal. Try using 'distance'.**")
                        if params['weights'] == 'distance':
                            st.markdown("**Inspect the Weights hyperparameter. Distance weights might not be optimal. Try using 'uniform'.**")
                else:
                    st.markdown(f':green[Hooray! We achieved our goal of over 80% accuracy on a classification model for {model_name}! 🤩  🙌  🎆  🎉 🎊 ]')
        
        # get the performance of all
        st.header('Model Performance Comparison')

        results = [rf_metrics, dt_metrics, kn_metrics]
        st.write('Performance metrics for all models:')
        st.write(results)
        plot_metrics(results, 'Accuracy')
        plot_metrics(results, 'Precision')
        plot_metrics(results, 'Recall')
        provide_feedback('Random Forest', rf_params, rf_metrics)
        provide_feedback('Decision Tree', dt_params, dt_metrics)
        provide_feedback('K Nearest Neighbors', kn_params, kn_metrics)