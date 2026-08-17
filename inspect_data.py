import joblib
import pandas as pd
from functools import reduce

# 1. File paths
file_paths = {
    "unemployment": "data/raw/coutries_unenmployement.csv",
    "inflation": "data/raw/inflation.csv",
    "oil_rents": "data/raw/oil_rents.csv",
    "gdp_growth": "data/raw/gdp_growth.csv",
    "exchange_rate": "data/raw/exchange_rate.csv",
}

id_cols = ['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code']


# 2. Select useful year columns before melting
def select_columns_to_keep(df, id_cols=None, min_non_null_ratio=1, verbose=True):
    if id_cols is None:
        id_cols = []

    # On garde seulement les colonnes d'années probables
    year_cols = [col for col in df.columns if str(col).isdigit()]
    value_cols = [col for col in year_cols if col not in id_cols]

    cols_to_keep = list(id_cols)

    for col in value_cols:
        non_null_ratio = df[col].notna().mean()
        if non_null_ratio >= min_non_null_ratio:
            cols_to_keep.append(col)
        else:
            if verbose:
                print(f"Drop column '{col}': non-null ratio = {non_null_ratio:.3f}")

    return cols_to_keep


# 3. Transform function
def transform_data(df, value_name):

    df = df.melt(
        id_vars=id_cols,
        var_name='Year',
        value_name='Value'
    )

  

    df = df.rename(columns={'Value': value_name})

    # Colonnes utiles pour merge / modélisation
    df = df[['Country Name', 'Country Code', 'Year', value_name]]

    # Nettoyage types
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df[value_name] = pd.to_numeric(df[value_name], errors='coerce')

    return df


# 4. Read, filter columns, transform
MAGHREB_COUNTRIES = ["DZA","MAR","TUN","EGY","LBY"]

def filter_maghreb(df, country_col='Country Code'):
    """
    Garde uniquement les lignes correspondant aux pays du Maghreb.
    """
    df_filtered = df[df[country_col].isin(MAGHREB_COUNTRIES)].copy()
    return df_filtered
dfs = {}

for name, path in file_paths.items():
    df_raw = pd.read_csv(path, skiprows=4)
    df_raw = filter_maghreb(df_raw, country_col='Country Code')
    cols_to_keep = select_columns_to_keep(
        df_raw,
        id_cols=id_cols,
        min_non_null_ratio=1.0,
        verbose=True
    )

    df_filtered = df_raw[cols_to_keep].copy()
    dfs[name] = transform_data(df_filtered, name)

    print(f"{name} transformed:")
    print("-" * 40)


# 5. Merge all indicators
df_merged = reduce(
    lambda left, right: pd.merge(
        left,
        right,
        on=['Country Name', 'Country Code', 'Year'],
        how='inner'
    ),
    dfs.values()
)

df_merged = df_merged.sort_values(['Country Code', 'Year']).reset_index(drop=True)



# 6. Optional save
df_merged.to_csv("data/raw/clean.csv", index=False)
print(df_merged['Year'].min(), df_merged['Year'].max(),df_merged.shape,df_merged.isna().sum().sum())



df_merged['inflation_next'] = df_merged.groupby('Country Code')['inflation'].shift(-1)
df_merged.to_csv("data/raw/clean.csv")

predict = df_merged[df_merged['inflation_next'].isna()]
predict.to_csv("data/raw/predict.csv", index=False)

print(predict.shape)
data = df_merged[df_merged['inflation_next'].notna()]
data.to_csv("data/raw/train.csv", index=False)

print(data.shape)
print(data['inflation_next'].isna().sum())
print(predict['Year'].unique())

train = data[(data['Year'] >= 1991) & (data['Year'] <= 2016)]
test = data[(data['Year'] >= 2017) & (data['Year'] <= 2020)]

mae = (test['inflation_next'] - test['inflation']).abs().mean()
print(f"MAE: {mae}")
X_train = train.drop(columns=['inflation_next', 'Country Name', 'Year', 'Country Code'])
y_train = train['inflation_next']
X_test = test.drop(columns=['inflation_next', 'Country Name', 'Year', 'Country Code'])
y_test = test['inflation_next']

print(X_train.shape)

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge

model = LinearRegression()
model.fit(X_train, y_train)       # ① apprend les coefficients — UNIQUEMENT sur train
preds = model.predict(X_test)     # ② prédit sur des lignes jamais vues (2017–2020)

print((preds - y_test).__abs__().mean())  # ③ compare les prédictions aux vraies valeurs

print("Ridge Regression")
ridge = Ridge(alpha=0.5)  # ① apprend les coefficients — UNIQUEMENT sur train
ridge.fit(X_train, y_train) 
preds_ridge = ridge.predict(X_test)
print((preds_ridge - y_test).__abs__().mean())  # ③
"""

data_encoded = pd.get_dummies(data, columns=['Country Code'])

mo
train = data_encoded[(data_encoded['Year'] >= 1991) & (data_encoded['Year'] <= 2016)]
test = data_encoded[(data_encoded['Year'] >= 2017) & (data_encoded['Year'] <= 2020)]  

X_train = train.drop(columns=['inflation_next', "Country Name"])
y_train = train['inflation_next']
X_test = test.drop(columns=['inflation_next', "Country Name" ])
y_test = test['inflation_next']

model = LinearRegression()
model.fit(X_train, y_train)       # ① apprend les coefficients — UNIQUEMENT sur train
preds = model.predict(X_test)     # ② prédit sur des lignes jamais vues (2017–2020)

print((preds - y_test).__abs__().mean())  # ③ compare les prédictions aux vraies valeurs
"""

poids = ({"model": model, "features": X_train.columns.tolist()})

joblib.dump(poids, "model.joblib")

saved = joblib.load("model.joblib")
print(saved["features"])
print(type(saved["model"]))

preds = saved["model"].predict(X_test[saved["features"]])
print((y_test - preds).__abs__().mean())