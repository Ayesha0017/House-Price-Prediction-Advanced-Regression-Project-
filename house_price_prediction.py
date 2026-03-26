import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV

df = pd.read_csv(r"C:\Users\ayesh\Downloads\house-prices-advanced-regression-techniques\train.csv")

df.drop(columns=['PoolQC', 'MSSubClass'], inplace=True)
cols_none = [
    'MiscFeature','Alley','Fence','FireplaceQu',
    'GarageType','GarageFinish','GarageQual','GarageCond',
    'BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',
    'MasVnrType'
]

for col in cols_none:
    df[col] = df[col].fillna("None")

cols_zero = [
    'GarageYrBlt','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF',
    'TotalBsmtSF','BsmtFullBath','BsmtHalfBath',
    'GarageCars','GarageArea'
]

for col in cols_zero:
    df[col] = df[col].fillna(0)

df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage']\
                     .transform(lambda x: x.fillna(x.median()))

cols_mode = [
    'MSZoning','Utilities','Functional',
    'Exterior1st','Exterior2nd','KitchenQual','SaleType'
]

for col in cols_mode:
    df[col] = df[col].fillna(df[col].mode()[0])

df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])

quality_features = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond']
qual_map = {'None': 0, 'NA': 0, 'Ex': 5, 'Gd': 4,'TA': 3, 'Fa': 2,'Po': 1}
for cols in quality_features:
    df[cols] = df[cols].map(qual_map)

bsmt_exp_map = {'None': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4}
df['BsmtExposure'] = df['BsmtExposure'].map(bsmt_exp_map)

bsmt_fin_map = {'None':0, 'Unf':1, 'LwQ':2, 'Rec':3, 'BLQ':4, 'ALQ':5, 'GLQ':6}
df['BsmtFinType1'] = df['BsmtFinType1'].map(bsmt_fin_map)
df['BsmtFinType2'] = df['BsmtFinType2'].map(bsmt_fin_map)

functional_map = {'Sal': 0, 'Sev': 1, 'Maj2': 2, 'Maj1': 3, 'Mod': 4, 'Min2': 5, 'Min1': 6, 'Typ': 7}
df['Functional'] = df['Functional'].map(functional_map)
garage_finish_map = {'None': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3}
df['GarageFinish'] = df['GarageFinish'].map(garage_finish_map)
paved_map = {'N': 0, 'P': 1, 'Y': 2}
df['PavedDrive'] = df['PavedDrive'].map(paved_map)
landslope_map = {'Sev': 0, 'Mod': 1,'Gtl': 2}
df['LandSlope'] = df['LandSlope'].map(landslope_map)
lotshape_map = {'IR3': 0, 'IR2': 1, 'IR1': 2,'Reg': 3, }
df['LotShape'] = df['LotShape'].map(lotshape_map)

categorical_cols = [
    'MSZoning', 'Street', 'Alley', 'LandContour', 'Utilities', 'LotConfig', 'Neighborhood', 'Condition1', 
    'Condition2', 'BldgType',
    'HouseStyle', 'RoofStyle', 'RoofMatl', 
    'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation', 'Heating', 'CentralAir', 'Electrical', 'GarageType', 'SaleType',
    'SaleCondition', 'Fence', 'MiscFeature'
]

df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

bool_cols = df.select_dtypes(include='bool').columns
df[bool_cols] = df[bool_cols].astype(int)

df.drop(columns = 'Id', inplace = True)

y = np.log1p(df['SalePrice'])
X = df.drop('SalePrice', axis=1)

cols_to_drop = [
    'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
    '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
    'YearBuilt', 'YearRemodAdd', 'YrSold',
    'FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath'
]

X = X.drop(columns=cols_to_drop)

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
vif["feature"] = X_train_scaled_df.columns
vif["VIF"] = [
    variance_inflation_factor(X_train_scaled_df.values, i)
    for i in range(X_train_scaled_df.shape[1])
]

vif = vif.sort_values(by="VIF", ascending=False)
print(vif.head(15))

model = RidgeCV(alphas=[0.1, 1, 10, 50, 100], cv=5)
model.fit(X_train_scaled_df, y_train)

y_pred_train = model.predict(X_train_scaled_df)
y_pred_test = model.predict(X_test_scaled_df)

print("Train R2:", r2_score(y_train, y_pred_train))
print("Test R2:", r2_score(y_test, y_pred_test))


y_test_actual = np.expm1(y_test)
y_pred_actual = np.expm1(y_pred_test)

rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
print("RMSE:", rmse)

lasso = LassoCV(cv=5)
lasso.fit(X_train_scaled_df, y_train)

y_pred_train = lasso.predict(X_train_scaled_df)
y_pred_test = lasso.predict(X_test_scaled_df)

print("Train R2:", r2_score(y_train, y_pred_train))
print("Test R2:", r2_score(y_test, y_pred_test))


y_test_actual = np.expm1(y_test)
y_pred_actual = np.expm1(y_pred_test)

rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))