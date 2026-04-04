import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from sklearn.linear_model import RidgeCV, LassoCV, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor


# =========================
# LOAD DATA
# =========================
def load_data(path):
    return pd.read_csv(path)


# =========================
# PREPROCESSING
# =========================
def preprocess(df):

    df = df.copy()

    # Drop columns
    df.drop(columns=['PoolQC', 'MSSubClass'], inplace=True)

    # Fill None-type
    cols_none = [
        'MiscFeature','Alley','Fence','FireplaceQu',
        'GarageType','GarageFinish','GarageQual','GarageCond',
        'BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',
        'MasVnrType'
    ]
    for col in cols_none:
        df[col] = df[col].fillna("None")

    # Fill zero
    cols_zero = [
        'GarageYrBlt','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF',
        'TotalBsmtSF','BsmtFullBath','BsmtHalfBath',
        'GarageCars','GarageArea'
    ]
    for col in cols_zero:
        df[col] = df[col].fillna(0)

    # LotFrontage
    df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage']\
                         .transform(lambda x: x.fillna(x.median()))

    # Mode fill
    cols_mode = [
        'MSZoning','Utilities','Functional',
        'Exterior1st','Exterior2nd','KitchenQual','SaleType'
    ]
    for col in cols_mode:
        df[col] = df[col].fillna(df[col].mode()[0])

    df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])

    # Ordinal mappings
    qual_map = {'None': 0, 'NA': 0, 'Ex': 5, 'Gd': 4,'TA': 3, 'Fa': 2,'Po': 1}
    quality_features = ['ExterQual','ExterCond','BsmtQual','BsmtCond','HeatingQC','KitchenQual','FireplaceQu','GarageQual','GarageCond']
    for col in quality_features:
        df[col] = df[col].map(qual_map)

    df['BsmtExposure'] = df['BsmtExposure'].map({'None':0,'No':1,'Mn':2,'Av':3,'Gd':4})
    df['BsmtFinType1'] = df['BsmtFinType1'].map({'None':0,'Unf':1,'LwQ':2,'Rec':3,'BLQ':4,'ALQ':5,'GLQ':6})
    df['BsmtFinType2'] = df['BsmtFinType2'].map({'None':0,'Unf':1,'LwQ':2,'Rec':3,'BLQ':4,'ALQ':5,'GLQ':6})

    df['Functional'] = df['Functional'].map({'Sal':0,'Sev':1,'Maj2':2,'Maj1':3,'Mod':4,'Min2':5,'Min1':6,'Typ':7})
    df['GarageFinish'] = df['GarageFinish'].map({'None':0,'Unf':1,'RFn':2,'Fin':3})
    df['PavedDrive'] = df['PavedDrive'].map({'N':0,'P':1,'Y':2})
    df['LandSlope'] = df['LandSlope'].map({'Sev':0,'Mod':1,'Gtl':2})
    df['LotShape'] = df['LotShape'].map({'IR3':0,'IR2':1,'IR1':2,'Reg':3})

    # One-hot
    categorical_cols = [
        'MSZoning','Street','Alley','LandContour','Utilities','LotConfig','Neighborhood',
        'Condition1','Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl',
        'Exterior1st','Exterior2nd','MasVnrType','Foundation','Heating','CentralAir',
        'Electrical','GarageType','SaleType','SaleCondition','Fence','MiscFeature'
    ]

    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Bool → int
    bool_cols = df.select_dtypes(include='bool').columns
    df[bool_cols] = df[bool_cols].astype(int)

    # Drop Id
    df.drop(columns='Id', inplace=True)

    return df


# =========================
# FEATURE ENGINEERING
# =========================
def prepare_features(df):

    y = np.log1p(df['SalePrice'])
    X = df.drop('SalePrice', axis=1)

    cols_to_drop = [
        'BsmtFinSF1','BsmtFinSF2','BsmtUnfSF',
        '1stFlrSF','2ndFlrSF','LowQualFinSF',
        'YearBuilt','YearRemodAdd','YrSold',
        'FullBath','HalfBath','BsmtFullBath','BsmtHalfBath'
    ]

    X = X.drop(columns=cols_to_drop)

    return X, y


# =========================
# METRICS
# =========================
def get_metrics(y_true, y_pred):
    y_true_actual = np.expm1(y_true)
    y_pred_actual = np.expm1(y_pred)

    rmse = np.sqrt(mean_squared_error(y_true_actual, y_pred_actual))
    mae = mean_absolute_error(y_true_actual, y_pred_actual)

    return rmse, mae


# =========================
# TRAIN BEST MODEL (CATBOOST)
# =========================
def train_catboost(X_train, y_train):

    model = CatBoostRegressor(random_state=42, verbose=0)

    param_grid = {
        'depth': [4, 6, 8],
        'learning_rate': [0.03, 0.05, 0.1],
        'iterations': [300, 500]
    }

    grid = GridSearchCV(model, param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid.fit(X_train, y_train)

    return grid.best_estimator_


# =========================
# MAIN PIPELINE
# =========================
def run_pipeline(path):

    df = load_data(path)
    df = preprocess(df)
    X, y = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    best_model = train_catboost(X_train, y_train)

    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)

    print("Train R2:", r2_score(y_train, y_pred_train))
    print("Test R2:", r2_score(y_test, y_pred_test))

    rmse, mae = get_metrics(y_test, y_pred_test)
    print("RMSE:", rmse)
    print("MAE:", mae)

    return best_model


# =========================
# RUN
# =========================
if __name__ == "__main__":
    model = run_pipeline("train.csv")