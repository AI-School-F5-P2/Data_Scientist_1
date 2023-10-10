import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.impute import KNNImputer

class CategoricalBmi(BaseEstimator, TransformerMixin):
    '''
    This class is a Custom Transformer that creates a new column
    based on the bmi values. It is then used inside the preprocessor
    pipeline.
    '''
    def __init__(self):
        #assign the column number of bmi which is 12 after transformations
        self.index_bmi = 12
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):

        column_num_bmi = X[:, self.index_bmi]

        #empty list to store new values created
        categorical_bmi = []
        
        for num_bmi in column_num_bmi:
            
            #conditional logic to create the new column
            #'Low'
            if num_bmi < 18.5:
                categorical_bmi.append(1)
            #'Normal'
            elif 18.5 <= num_bmi < 24.9:
                categorical_bmi.append(2)
            #'Overweight'
            elif 25 <= num_bmi < 29.9:
                categorical_bmi.append(3)
            #'Mild Obesity'
            elif 30 <= num_bmi < 34.9:
                categorical_bmi.append(4)
            #'Moderate Obesity'
            elif 35 <= num_bmi < 39.9:
                categorical_bmi.append(5)
            #'Severe Obesity'
            else:
                categorical_bmi.append(6)
        
        X = np.delete(X, self.index_bmi, axis=1)
        
        return np.c_[X, categorical_bmi]


def impute_smokers_age(X_train):
    '''
    This functions gets the matrix X and returns the same matrix with 
    some changes in the smoking_status column:
    children under 12 years old -> 'never smoked'
    unknown -> nulls (np.nan) for later imputation.
    '''
    min_age = 12

    X_train.loc[(X_train['age'] <= min_age) & (X_train['smoking_status'] == 'Unknown'), 'smoking_status'] = 'never smoked'
    X_train.loc[(X_train['smoking_status'] == 'Unknown'), 'smoking_status'] = np.nan
    
    return X_train


def create_preprocessor(X_train):
    '''
    This function creates the preprocessor to transform the X matrix.
    It returns the object preprocessor.
    '''
    func_trans = FunctionTransformer(impute_smokers_age)

    ordinal_columns = ['gender', 'ever_married', 'Residence_type', 'smoking_status']
    onehot_columns = ['work_type']
    std_columns = ['age', 'avg_glucose_level']

    col_transformer = ColumnTransformer([('ordinal', OrdinalEncoder(), ordinal_columns), 
                                         ('onehot', OneHotEncoder(), onehot_columns),
                                         ('std', StandardScaler(), std_columns)], remainder='passthrough')
    
    preprocessor = Pipeline([('nulls', func_trans), ('scale_code', col_transformer), 
                             ('knn', KNNImputer()), ('bmi_to_cat', CategoricalBmi())])
    
    X_train_transformed = preprocessor.fit_transform(X_train)

    return preprocessor, X_train_transformed