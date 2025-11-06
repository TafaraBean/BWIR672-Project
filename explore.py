#get the variable types of gpd_f.csv
import pandas as pd
df = pd.read_csv("gpd_f.csv")
print(df.dtypes)

#print head of gpd_f.csv
print(df.head())

#get the distinct values of the _PartInd_ column
print(df["_PartInd_"].unique())

#the count of times each value appears in the _PartInd_ column
print(df["_PartInd_"].value_counts())

#get the train_df which is when _PartInd_ == 0
train_df = df.loc[df["_PartInd_"] == 1].reset_index(drop=True)
len_train = len(train_df)
print(f"Length of train_df: {len_train}")

#let xi_mle and sig_mle be equal to 1 if in the training and validation set respectively
df.loc[df["_PartInd_"] == 1, "xi_mle"] = 1
df.loc[df["_PartInd_"] == 1, "sig_mle"] = 1
df.loc[df["_PartInd_"] == 0, "sig_mle"] = 1
df.loc[df["_PartInd_"] == 0, "xi_mle"] = 1

print(df.head())
print(df.tail())

FEAT_COLS = [f"Q{i}" for i in range(1, 12)]
TGT_COLS  = ["xi", "sig"]
needed = ["_PartInd_","N"]  + FEAT_COLS + TGT_COLS + ["iqr","xi_mle","sig_mle"]
df = df[needed].dropna().copy()

print(df.dtypes)

train_df = df.loc[df["_PartInd_"] == 1].reset_index(drop=True)
len_train = len(train_df)
print(f"Length of train_df: {len_train}")