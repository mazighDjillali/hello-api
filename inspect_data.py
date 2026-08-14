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

def filter_maghreb(df, country_col='Country Name'):
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
    print(dfs[name].head())
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

print(df_merged.head())


# 6. Optional save
df_merged.to_csv("data/raw/merged_indicators.csv", index=False)
print(df_merged['Year'].min(), df_merged['Year'].max(),df_merged.shape,df_merged.isna().sum().sum())