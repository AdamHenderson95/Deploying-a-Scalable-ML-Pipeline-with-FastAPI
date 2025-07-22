import pandas as pd

df = pd.read_csv('census.csv', index_col=0)

print(df.shape)
print(df.info)