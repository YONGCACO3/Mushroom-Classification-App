import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.metrics import precision_score, recall_score 

st.set_page_config(
    page_title="Mushroom Classification App",
    page_icon="🍄",
    initial_sidebar_state="expanded",
)


def main():
    st.title("Mushroom  Classification Web App")
    st.sidebar.title("Mushroom  Classification Web App")   
    st.markdown("Are your mushrooms edible or poisonous🍄👾?")  
    st.sidebar.markdown("Are your mushrooms edible or poisonous🍄👾?")
    default_image_url = "https://image.freepik.com/free-vector/mushroom-anatomy-labeled-biology-illustration_1995-566.jpg"
    st.image(default_image_url, caption="Mushroom Anatomy", use_column_width=True)

    @st.cache_data(persist=True)
    def load_data():
        data = pd.read_csv('mushrooms.csv')
        label = LabelEncoder()
        # Iterate through each column in the DataFrame and use the 
        # LabelEncoder to transform categorical data in each column to numerical values
        for col in data.columns:
            data[col] = label.fit_transform(data[col])
        return data
    
    @st.cache_data(persist=True)
    def split(df):
        y = df.type
        x = df.drop(['type'],axis=1)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=27)
        return x_train, x_test, y_train, y_test
    
    def plot_metrics(metrics_list, model, x_test, y_test, class_names):
    # First row, single plot
        if 'Confusion Matrix' in metrics_list:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, model.predict(x_test))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
            fig, ax = plt.subplots(figsize=(6, 4))
            disp.plot(ax=ax)
            st.pyplot(fig)

        # Second row, two plots stacked horizontally
        if 'ROC Curve' in metrics_list or 'Precision-Recall Curve' in metrics_list:
            st.subheader("Performance Metrics")
            col1, col2 = st.columns(2)

            if 'ROC Curve' in metrics_list:
                col1.subheader("ROC Curve")
                fpr, tpr, _ = roc_curve(y_test, model.predict_proba(x_test)[:, 1])
                roc_auc = auc(fpr, tpr)
                disp_roc = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
                fig_roc, ax_roc = plt.subplots()
                disp_roc.plot(ax=ax_roc)
                col1.pyplot(fig_roc)

            if 'Precision-Recall Curve' in metrics_list:
                col2.subheader("Precision-Recall Curve")
                precision, recall, _ = precision_recall_curve(y_test, model.predict_proba(x_test)[:, 1])
                disp_pr = PrecisionRecallDisplay(precision=precision, recall=recall)
                fig_pr, ax_pr = plt.subplots()
                disp_pr.plot(ax=ax_pr)
                col2.pyplot(fig_pr)


    
    df = load_data()
    x_train, x_test, y_train, y_test = split(df)
    labels = ['Edible', 'Poisonous']
    if st.sidebar.checkbox("Show raw data", False):
        st.subheader("Mushroom Data Set (Classification)")
        st.markdown("This dataset describes 23 species of gilled mushrooms from the Agaricus and Lepiota family. The samples are hypothetical and sourced from The Audubon Society Field Guide to North American Mushrooms (1981). Each species is labeled as definitely edible, definitely poisonous, or of unknown edibility (not recommended), with the latter combined with the poisonous class.") 
        st.write(df)

    st.sidebar.subheader("Choose Classifier")
    classifier = st.sidebar.selectbox("Classifier", ("Logistic Regression","Support Vector Machine (SVM)",  "Random Forest"))

    

    if classifier == "Logistic Regression":
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C_LR')
        max_iter = st.sidebar.slider("Maximum number of iterations", 100, 500, key='max_iter')
        metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Logistic Regression Results")
            model = LogisticRegression(C=C, max_iter=max_iter)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, labels=labels).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels=labels).round(2))
            plot_metrics(metrics, model, x_test, y_test, labels)

    if classifier == "Support Vector Machine (SVM)": 
        st.sidebar.subheader("Model Hyperparameters")
        # choose parameters for the model
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C')
        kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), key='kernel')
        gamma = st.sidebar.radio("Gamma (Kernel Coefficient)", ("scale", "auto"), key='gamma')
        #class_weight = st.sidebar.selectbox("Class Weight", ["None", "balanced"], key='class_weight')

        metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Support Vector Machine (SVM) Results")
            model = SVC(C=C, kernel=kernel, gamma=gamma, probability=True)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, labels=labels).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels=labels).round(2))
            plot_metrics(metrics, model, x_test, y_test, labels)
    
    if classifier == "Random Forest":
        st.sidebar.subheader("Model Hyperparameters")
        n_estimators = st.sidebar.number_input("The number of trees in the forest", 100, 5000, step=10, key='n_estimators')
        max_depth = st.sidebar.number_input("The maximum depth of the tree", 1, 20, step=1, key='max_depth')
        bootstrap = st.sidebar.radio("Bootstrap samples when building trees", (True, False), key='bootstrap')

        metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Random Forest Results")
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, labels=labels).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels=labels).round(2))
            plot_metrics(metrics, model, x_test, y_test, labels)

            

     



if __name__ == '__main__':
    main()


