
from typing import List
import numpy as np  # for mathematical operations
import pandas as pd  # to work with dataframes
# import seaborn as sns
from sklearn.decomposition import randomized_svd
import streamlit as st
# from matplotlib import pyplot as plt
from sklearn.ensemble import (
    
    RandomForestClassifier
)
from sklearn.metrics import (

    f1_score,
    precision_score, recall_score, roc_auc_score
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

RANDOM_STATE = 1

# multiple containers to hold differnt sub sections
header = st.container()
dataset = st.container()
eda = st.container()
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

        f1: float = float(f1_score(y_test, pred))
        f1_score_c.append(f1)

        # roc = roc_auc_score(y_test, pred)
        # roc_auc.append(roc)

        # prec = precision_score(y_test, pred)
        # precision.append(prec)

        # rec = recall_score(y_test, pred)
        # recall.append(rec)

    return np.mean(f1_score_c)


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


with header:
    st.title('Welcome to my Data Science Project')


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

    credit_card_data = credit_card_data.replace(
        {'Existing Customer': 0, 'Attrited Customer': 1})

    # category conversion
    cc_df = cate_type(credit_card_data)


with eda:
    st.header('Explanatory Data analysis')
    st.text('I have performed statistical analysis on numerical features.')
    st.text('I have plotted frequency charts, scatter plot for dataset features.')
    st.write(
        'Link to the Github repository [click here](https://github.com/abhishek-mehra/Credit_Card)')


# 1. asking user for feature engineering options :(a) create a new feature  (b) scale the features

# 2. asking user for Model selection: (a) Random Forest (b) XGBoost

# 3. asking user for parameters selection: (a)number of estimators (b) max depth


with machine_learning:
    st.header('Machine learning: Predicting Attrition')
    st.subheader('Data Peparation Options')

    # (a)asking user for feature engineering options :(a) create a new feature
    feature_selection_ouput = st.selectbox(
        'Do you want to add a new feature?', ('Yes', 'No'))

    # (b) scale the features
    scale_selection_ouput = st.selectbox(
        'Do you want to scale the numerical features?', ('Yes', 'No'))

    if feature_selection_ouput == 'Yes':
        cc_df = new_feat(cc_df)

    if scale_selection_ouput == 'Yes':
        cc_df = stan_scal(cc_df)

    st.write("Data sample:", cc_df.head())

    st.subheader('ML Experiment Options')

    # 2. asking user for Model selection: (a) Random Forest
    model_selection_ouput = st.selectbox(
        'Which model do you want to select ?', ('Random Forest Classifier', 'XGBClassifier'))

    # 3.asking user for Model selection: (a)number of estimators
    estimators_input = st.slider(
        'What should be the number of trees?', min_value=100, max_value=600, step=100)

    # 3 asking user for max depth
    max_depth_input = st.slider(
        'What should be the max depth of trees?', min_value=2, max_value=8, step=1)

    # 4 asking user for cv folds
    n_folds = st.slider('How many CV folds?',
                        min_value=5, max_value=10, step=1)

    if model_selection_ouput == 'Random Forest Classifier':
        rf = RandomForestClassifier(
            n_estimators=estimators_input, max_depth=max_depth_input, random_state=RANDOM_STATE)
        st.write("Average CV F1 score is: ", round(
            cv_score_model(cc_df, rf, n_folds), 3))

    else:
        xg = XGBClassifier(n_estimators=estimators_input,
                           max_depth=max_depth_input, random_state=RANDOM_STATE)
        st.write("Average CV F1 score is: ", round(
            cv_score_model(cc_df, xg, n_folds), 3))
