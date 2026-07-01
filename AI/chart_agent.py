# import matplotlib.pyplot as plt


# def create_chart(df):

#     numeric=df.select_dtypes("number")

#     numeric.sum().plot()

#     plt.savefig("chart.png")

#     return "chart.png"



import os
import matplotlib.pyplot as plt


def create_charts(df):

    os.makedirs(
        "reports/charts",
        exist_ok=True
    )

    numeric_df = df.select_dtypes(
        include=["number"]
    )

    if len(numeric_df.columns) < 2:

        raise Exception(
            "CSV must contain at least 2 numeric columns"
        )

    chart1 = (
        "reports/charts/chart1.png"
    )

    chart2 = (
        "reports/charts/chart2.png"
    )

    # Chart 1

    plt.figure(
        figsize=(6, 4)
    )

    numeric_df[
        numeric_df.columns[0]
    ].hist()

    plt.title(
        numeric_df.columns[0]
    )

    plt.savefig(
        chart1
    )

    plt.close()

    # Chart 2

    plt.figure(
        figsize=(6, 4)
    )

    numeric_df[
        numeric_df.columns[1]
    ].hist()

    plt.title(
        numeric_df.columns[1]
    )

    plt.savefig(
        chart2
    )

    plt.close()

    return [

        chart1,

        chart2
    ]