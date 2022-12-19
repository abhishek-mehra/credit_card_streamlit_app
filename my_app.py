
from typing import List
import numpy as np  # for mathematical operations
import pandas as pd  # to work with dataframes
# import seaborn as sns
from sklearn.model_selection import train_test_split

import streamlit as st
# from matplotlib import pyplot as plt
from sklearn.ensemble import (

    RandomForestClassifier
)
from sklearn.metrics import (

    f1_score,
    precision_score, recall_score, classification_report, roc_curve
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier

import seaborn as sns
import matplotlib.pyplot as plt

RANDOM_STATE = 1

# multiple containers to hold differnt sub sections
header = st.container()
dataset = st.container()
eda = st.container()
data_preparation = st.container()
machine_learning = st.container()


# Helper FUNCTIONS

# categorical conversion
def cate_type(df) -> pd.DataFrame:
    ndf: pd.DataFrame = df.copy(deep=True)
    # selecting string type columns
    c: pd.Index = ndf.select_dtypes(include='object').columns
    for i in c:
        # categorical columns to panda category type
        ndf[i] = ndf[i].astype('category').cat.codes
    return ndf


# Cross Validation
@st.cache
def cv_score_model(df, model, folds=5, label_col_name="Attrition_Flag"):
    y = df[label_col_name].values  # dataframe to numpy array
    # dataframe to numpy array
    x = df.drop(label_col_name, axis=1, inplace=False).values

    # creating object of StratifiedKFold class
    skfold = StratifiedKFold(
        random_state=RANDOM_STATE,
        n_splits=folds, shuffle=True)

    # initialzing empty list to store f1 scores of cross validation folds
    f1_score_c: List[float] = []
    roc_auc: List[float] = []
    precision: List[float] = []
    recall: List[float] = []

    # skfold.split returns the indices
    for train_i, test_i in skfold.split(x, y):
        X_train = x[train_i]
        y_train = y[train_i]

        X_test = x[test_i]
        y_test = y[test_i]

        model.fit(X_train, y_train)

        pred = model.predict(X_test)

        # predicting probabilty of predictions
        pred_prob = model.predict_proba(X_test)[::, 1]
        fnr, tpr, _ = roc_curve(y_test, pred_prob)

        f1: float = float(f1_score(y_test, pred))
        f1_score_c.append(f1)

        # roc = roc_auc_score(y_test, pred)
        # roc_auc.append(roc)

        prec = precision_score(y_test, pred)
        precision.append(prec)

        rec = recall_score(y_test, pred)
        recall.append(rec)
    
    output_dictionary = {'f1_score':np.mean(f1_score_c),
    'precision':np.mean(precision),
    'recall':np.mean(recall),
    'fnr':fnr,
    'tpr':tpr
    }


    return output_dictionary

# simple evaluation


def train_eval(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # # miss classified data - false positives
    # missclassified = y_test != predictions
    # miss_classified_df = X_test[missclassified]

    #dataframe containing just y_test and predictions
    output_df = pd.DataFrame({'Actual':y_test, 'Predicted':predictions})

    # miss classified data - false positives
    fp_df = X_test.loc[(output_df['Actual']==0) & (output_df['Predicted']==1)]
    

    # miss classified data - false negatives
    fn_df = X_test.loc[(output_df['Actual']==1) & (output_df['Predicted']==0)]
    
    output_dic = {'f1':round(f1_score(y_test, predictions), 3),
    'precision':round(precision_score(y_test, predictions), 3),
    'recall':round(recall_score(y_test, predictions), 3),
    'classification':classification_report(y_test, predictions, output_dict=True),
    'false_positves':fp_df,
    'false_negatives':fn_df}

    return output_dic

# adding a new feature
@st.cache
def new_feat(df):
    df['avg_trans'] = df['Total_Trans_Amt']/df['Total_Trans_Ct']
    return df


# scaling
@st.cache
def stan_scal(df):
    # Dropping label column
    ndf = df.drop(['Attrition_Flag'], axis=1, inplace=False)

    # Split
    df_category = ndf.select_dtypes(include=['int8'])
    numerical_df = ndf.select_dtypes(include=['float64', 'int64'])

    # Sanity test
    assert len(df_category.columns) + \
        len(numerical_df.columns) == len(ndf.columns)

    # Scale numerical columns - ndarray
    stan_sc = StandardScaler()
    stan_sc.fit(numerical_df)

    # create numerical dataframe from ndarray
    df_numer = pd.DataFrame(columns=stan_sc.feature_names_in_,
                            data=stan_sc.transform(numerical_df))

    # create final dataframe by joining scaled numerical features, categorical features and label
    df_final = df_category.join(df_numer)
    df_final['Attrition_Flag'] = df['Attrition_Flag']

    # sanity tests
    assert len(df_final.columns) == len(df.columns)
    assert set(df_final.columns) == set(df.columns)

    return df_final


# train test split
@st.cache
def ttsplit(df2, label_col_name='Attrition_Flag', test_size=0.2):
    y = df2[label_col_name]
    df2 = df2.drop(label_col_name, axis=1, inplace=False)
    assert label_col_name not in df2.columns

    X_train, X_test, y_train, y_test = train_test_split(
        df2, y, test_size=test_size, shuffle=True, random_state=RANDOM_STATE, stratify=y)

    return X_train, X_test, y_train, y_test


with header:
    st.title('Welcome to my Data Science Project')
    st.write('[GitHub Repo](https://github.com/abhishek-mehra/Credit_Card)')
    st.write('[LinkedIn](https://www.linkedin.com/in/mehra-abhishek/)')

with dataset:
    st.header('It is a Credit card customers dataset')
    st.write(
        'The source of the data is kaggle:[Credit card customers](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers)')

    # input data

    credit_card_data = pd.read_csv(
        "./data/BankChurners.csv")
    st.write(credit_card_data.head(5))
    credit_card_data.drop(['Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1',
                           'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2',
                           'CLIENTNUM'], axis=1,
                          inplace=True)
    # category conversion
    cc_df = cate_type(credit_card_data)

#feature importance function
def fi(model):
    fi_df = pd.DataFrame({'Features': model.feature_names_in_,
                     'Importance': model.feature_importances_})
    fi_df.sort_values(by='Importance', ascending=False, inplace=True)
    fig3 = plt.figure(figsize=(10,4))
    plt.xticks(rotation='vertical')
    sns.barplot(data=fi_df,x=fi_df['Features'],y=fi_df['Importance'])
    
    
    return fig3



with eda:
    st.header('Exploratory Data analysis')
    st.markdown('The dataset contains information about the customer, including their age, gender, income bracket, and credit card characteristics like their total revolving debt, credit limit, months of inactivity, and open to buy etc. The dependent variable is attrition which tells us whether the customer is still associated with the services or has left the credit card service.')
    st.markdown('I conducted additional analysis on the dataset attributes. The data visualizations reveal several significant observations.')

    fig = plt.figure(figsize=(10, 4))
    sns.scatterplot(data=credit_card_data, x='Total_Trans_Amt',
                    y='Total_Trans_Ct', hue='Attrition_Flag')
    st.pyplot(fig)

    fig2 = plt.figure(figsize=(10, 4))
    sns.scatterplot(data=credit_card_data, x='Avg_Utilization_Ratio',
                    y='Total_Revolving_Bal', hue='Attrition_Flag')
    st.pyplot(fig2)

    st.markdown('Customers with higher transaction amounts and higher transaction numbers have a lower attrition rate than those with lower transaction amounts and lower transaction counts. In the second scatter plot, the number of attrited clients is concentrated in the bottom half, where credit is used the least. As a result, users who use the service less frequently abandon it.')


# 1. asking user for feature engineering options :(a) create a new feature  (b) scale the features

# 2. asking user for Model selection: (a) Random Forest (b) XGBoost

# 3. asking user for parameters selection: (a)number of estimators (b) max depth


# replacing label class as 0 and 1.
    credit_card_data = credit_card_data.replace(
        {'Existing Customer': 0, 'Attrited Customer': 1})

def exec_data_prep(cc_df, feature_selection_ouput, scale_selection_ouput):
    if feature_selection_ouput == 'Yes':
        cc_df = new_feat(cc_df)

    if scale_selection_ouput == 'Yes':
        cc_df = stan_scal(cc_df)
    
    return cc_df


with data_preparation:
    st.subheader('Data Peparation Options')

    
    # (a)asking user for feature engineering options :(a) create a new feature
    # st.markdown('Adding this feature will give more insights to machine learning model')
    # st.markdown(
        # 'Average transaction amount = Total transaction amount/ Total transaction count')

    form_data_prep = st.form(key = 'data_prep')
    
    
    feature_selection_ouput = form_data_prep.selectbox('Do you want to add a new feature, average transaction amount.', ('Yes', 'No'))

    # (b) scale the features
    st.markdown('Scaling will bring all the numerical features on the same scale, helping the machine learning model to learn better.')
    scale_selection_ouput = form_data_prep.selectbox(
        'Do you want to scale the numerical features?', ('Yes', 'No'))

    data_prep_button_pressed = form_data_prep.form_submit_button("Click for data prep")

    if data_prep_button_pressed:
        cc_df = exec_data_prep(cc_df, feature_selection_ouput, scale_selection_ouput)
        st.write("Data sample:", cc_df.head())

with machine_learning:
    machine_learning.header('Machine learning: Predicting Attrition')
    machine_learning.subheader('ML Experiment Options')

    form_ml = st.form(key = 'ml')
    # 2. asking user for Model selection: (a) Random Forest
    model_selection_ouput = form_ml.selectbox(
        'Which model do you want to select ?', ('XGBClassifier', 'Random Forest Classifier'))

    # 3.asking user for Model selection: (a)number of estimators
    estimators_input = form_ml.slider(
        'What should be the number of trees?', min_value=100, max_value=600, step=100)

    # 3 asking user for max depth
    max_depth_input = form_ml.slider(
        'What should be the max depth of trees?', min_value=2, max_value=8, step=1)

    # 4 asking user for cv folds
    n_folds = form_ml.slider('How many CV folds?',
                        min_value=5, max_value=10, step=1)


    ml_form_submit_button_output = form_ml.form_submit_button("Submit for training and evaluation")

    # Dividing data into train and test
    # we are divinding dataset into  train -90% and test -10% of the dataset
    X_train, X_test_f, y_train, y_test_f = ttsplit(cc_df, test_size=0.2)
    # this is required since my cv function requires complete data set. features as well as label
    train_final = X_train.join(y_train)


    if ml_form_submit_button_output:
        if model_selection_ouput == 'Random Forest Classifier':
            model = RandomForestClassifier(
                n_estimators=estimators_input, max_depth=max_depth_input, random_state=RANDOM_STATE, n_jobs = -1)
        else:
            model = XGBClassifier(n_estimators=estimators_input,
                                max_depth=max_depth_input, random_state=RANDOM_STATE, n_jobs = -1)

        st.subheader('Evaluation Metrics')

        # results from cv evaluation stored in a list. F1, precision, recall in this order.
    

        result_train_dic = cv_score_model(train_final, model, n_folds)


        tpr = result_train_dic['tpr']
        fnr = result_train_dic['fnr']
        result_train = ['Train set'] + list([result_train_dic['f1_score'],result_train_dic['precision'],result_train_dic['recall']])

        # printing roc-auc
        fig = plt.figure(figsize=(10, 4))
        plt.plot(fnr, tpr)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')

        # result of evaluation from testing set
        result_test_dic = train_eval(model, X_train, X_test_f, y_train, y_test_f)

    

        # miss classified data
        fp_df = result_test_dic['false_positves']
        fn_df = result_test_dic['false_negatives']


        # putting classification report in a dataframe
        classification_report_output = pd.DataFrame(
            data=result_test_dic['classification']).transpose()

        classification_report_output = classification_report_output.rename(
            index={'0': 'Existing Customer', '1': 'Attrited Customer'})

        classification_report_output = classification_report_output.round(3)

        # adding a column name

        result_test = ['Hold out test set'] + list([result_test_dic['f1'],result_test_dic['precision'],result_test_dic['recall']])

        # adding rows to the dataframe using loc(key)
        outputs = pd.DataFrame(
            columns=['Scores', 'F1 ', 'Precision ', 'Recall score'])
        outputs.loc[0] = result_train
        outputs.loc[1] = result_test

    
        
    
        
        # showing the ouputs
        st.write(outputs.head())
        st.subheader('Classification Report on Hold out set')
        st.write(classification_report_output)

        st.subheader('Miss-classified data ')
        st.subheader('Data wrongly classified as positive by the model')
        st.write(fp_df)

        st.subheader('Data wrongly classified as negative by the model')
        st.write(fn_df)

        st.subheader('Receiver operating characteristics curve on Cross validated dataset')
        st.pyplot(fig)


      
        fig = fi(model)
        st.pyplot(fig)
        # 

   

 