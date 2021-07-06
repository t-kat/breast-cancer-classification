import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


def read_in(datafile):
    """
    Function for reading in input file
    :param
            datafile: txt, csv file
    :return:
            X: list of lists, independent variables
            Y: list, dependent variable
    """
    df = pd.read_csv(datafile)
    # Binarize target column
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0}).astype(int)
    # Define dependent and independent variable columns
    Y = df.values[:, 1].astype(int)
    X = df.values[:, 2:].astype(float)

    return X, Y

def standardize_X(indep):
    """
    Function for applying Z-score standardization on feature variables.
    :param indep: list of lists, independent variables
    :return:
    """
    # Standardize
    scaler = StandardScaler()
    indep = scaler.fit_transform(indep)

    return indep



def nn_model():
    """
    Function for creating rectified neural network model with one hidden layer.

    model creation using logarithmic loss function with Adam optimization for gradient descent.
    :param:
            nr_input_variables: int, defines number of neurons
    :return:
            model: neural network model
    """
    model = Sequential()
    # Apply rectified linear activation unit to hidden layers
    model.add(Dense(30, input_dim=30, activation='relu'))
    # Apply sigmoid activation function to output layer
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def cross_validate(X, Y):
    """
    Function for model evaluation.

    :param X: list of list, independent variables
    :param Y: list, dependent variable
    :return:
            avg_res: average cross validation result
            std_res: standard deviation of cross validation result
    """
    estimator = KerasClassifier(build_fn=nn_model, epochs=200, batch_size=10, verbose=0)
    kfold = StratifiedKFold(n_splits=10, shuffle=True)
    results = cross_val_score(estimator, X, Y, cv=kfold)
    avg_res = results.mean() * 100
    std_res = results.std() * 100

    return avg_res, std_res




if __name__ == '__main__':
    # Get data
    X, Y = read_in('data.csv')

    # Standardize X
    X_standard = standardize_X(X)

    # Build function, number of independent variables as input
    build_func = nn_model()

    # Cross validation
    avg_result, std_result = cross_validate(X_standard,Y)
    print(avg_result)
