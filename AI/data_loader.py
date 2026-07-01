# # utils/data_loader.py

# import pandas as pd


# def load_data(uploaded_file):

#     try:

#         df = pd.read_csv(
#             uploaded_file
#         )

#         return df

#     except Exception as e:

#         return str(e)


# def clean_data(df):

#     df = df.drop_duplicates()

#     df = df.fillna("Unknown")

#     return df


# def dataset_summary(df):

#     summary = {

#         "Rows": df.shape[0],

#         "Columns": df.shape[1],

#         "Missing Values":

#         df.isnull().sum().to_dict(),

#         "Column Names":

#         list(df.columns)

#     }

#     return summary





# import pandas as pd


# def load_data(file):

#     if file.name.endswith(".csv"):

#         df = pd.read_csv(file)

#     elif file.name.endswith(".xlsx"):

#         df = pd.read_excel(file)

#     else:
#         raise Exception(
#             "Only CSV and XLSX supported"
#         )

#     return df




import pandas as pd


def load_data(file):

    try:

        # Reset pointer
        file.seek(0)

        df = pd.read_csv(file)

        return df

    except Exception as e:

        raise Exception(
            f"CSV Loading Error: {str(e)}"
        )