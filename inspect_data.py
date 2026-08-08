import pandas as pd


df = pd.read_csv('data/raw/coutries_unenmployement.csv',    skiprows=4,
)
df.head(5)
df.shape
df.columns.tolist()
df.info()
df.isna().sum()