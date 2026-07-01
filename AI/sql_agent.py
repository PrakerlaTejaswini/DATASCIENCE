# def analyze_data(df):

#     return {

#         "rows": df.shape[0],

#         "columns": df.shape[1],

#         "missing":

#         df.isnull().sum().to_dict(),

#         "summary":

#         df.describe().to_string()

#     }



import pandas as pd


def run_sql(

    df,

    query="describe"

):

    query = query.lower()

    if query == "describe":

        return df.describe()

    elif query == "count":

        return len(df)

    elif query == "columns":

        return list(df.columns)

    elif query == "missing":

        return df.isnull().sum()

    return df.head()