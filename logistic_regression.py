import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "browser"  # for opening the PCA figure in browser
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


def read_in(datafile):
    """
    Function for reading in input file
    :param datafile: csv file
    :return:
    """
    df = pd.read_csv(datafile)
    # replace 0s with NaNs
   # df = df.replace(0, np.nan)
    # Binarize target column
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0}).astype(int)

    return df


def is_unbalanced(cancer_data):
    """
    Function for counting occurrences of benign, malignant cancer types in the dataset.
    :param cancer_data: pandas df
    :return: str
    """
    b_count = data['diagnosis'].value_counts()[0]
    m_count = data['diagnosis'].value_counts()[1]

    if b_count / m_count * 100 or m_count / b_count * 100 != 50:
        if b_count / m_count * 100 or m_count / b_count * 100 != 60:
            return 'The imbalance in the dataset is less than 10%.'
    else:
        return 'The dataset is imbalanced - additional stratified sampling is required for accurate results.'


def is_na(cancer_data):
    return cancer_data.isna().sum()


def impute_values(cancer_data):
    # take a look at imputer from sklearn.preprocessing import Imputer
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputer = imputer.fit(cancer_data.iloc[:, 2:])
    cancer_data.iloc[:, 2:] = imputer.transform(cancer_data.iloc[:, 2:])

    return cancer_data


def split_standardize(cancer_data):
    """
    Function for splitting data into target and feature variables & applying Z-score standardization
    on feature variables.
    :param cancer_data:
    :return:
    """
    # Separate features
    X = cancer_data.iloc[:, 2:].values
    # Separate target
    diagnosis = cancer_data.loc[:, ['diagnosis']].values
    # Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, diagnosis


def perform_pca(cancer_data, X):
    """
    Function for reducing dataset dimensionality by performing PCA.
    :param cancer_data: pandas df
    :param X: array, contains only feature variable columns
    :return: PCA plot popup
    """
    # Covariance matrix
    covariance_mat = PCA(n_components=2)
    # Get principal components
    pcs = covariance_mat.fit_transform(X)

    # Grab only the first 2 PCs
    pca_res = pd.DataFrame(data=pcs[:, 0:2],
                           columns=['PC1', 'PC2'])
    # Add back diagnosis column
    pca_res = pd.concat([pca_res, cancer_data[['diagnosis']]], axis=1)
    # Make names pretty
    pca_res = pca_res.replace(to_replace='B', value='Benign')
    pca_res = pca_res.replace(to_replace='M', value='Malignant')

    # Get total variance explained by first 2 PCs
    total_var = covariance_mat.explained_variance_ratio_
    labels = {
        'PC' + str(i + 1): f"PC {i + 1} ({var:.1f}%)"
        for i, var in enumerate(total_var * 100)
    }

    # Plot
    fig = px.scatter(pca_res, x='PC1', y='PC2', color=pca_res['diagnosis'],
                     labels=labels)
    fig.update_layout(legend_title_text='Diagnosis')
    # set background transparent (a = alpha)
    fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
    })
    # black box around plot
    fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)

    fig.show()


def log_regression(X, diagnosis):
    # Split data 67-33% with stratification (M/B 212/357)
    X_train, X_test, Y_train, Y_test = train_test_split(X, diagnosis, test_size=0.33, random_state=0,
                                                        stratify=diagnosis)

    # Model
    model = LogisticRegression()
    # Model fit with changing the Y_train shape into array
    fit = model.fit(X_train, Y_train.ravel())
    # Make predictions
    pred_res = model.predict(X_test)

    # Confusion matrix
    con_mat = metrics.confusion_matrix(Y_test, pred_res)

    # Calculate accuracy, precision, recall
    acc = metrics.accuracy_score(Y_test, pred_res) * 100
    prec = metrics.precision_score(Y_test, pred_res) * 100
    recall = metrics.recall_score(Y_test, pred_res) * 100

    return [acc, prec, recall]


if __name__ == '__main__':
    # Get data
    data = read_in('data.csv')

    unbalance_check = is_unbalanced(data)
    print(unbalance_check)
    # Check for missing values
    # missing_values = is_na(data)

    # Impute missing values
    #full_data = impute_values(data)

    # Get features, response variable
    x, y = split_standardize(data)

    # Perform PCA
    # pca = perform_pca(data, x)

    # Logistic regression
    model_result = log_regression(x, y)
    print("Model accuracy:", round(model_result[0], 2))
    print("Model precision:", round(model_result[1], 2))
    print("Model recall:", round(model_result[2], 2))
