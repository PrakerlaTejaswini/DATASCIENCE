# import os
# import matplotlib.pyplot as plt
# from PIL import Image


# def build_dashboard():

#     os.makedirs(
#         "reports/dashboards",
#         exist_ok=True
#     )

#     dashboard = (
#         "reports/dashboards/dashboard.png"
#     )

#     chart1 = (
#         "reports/charts/chart1.png"
#     )

#     chart2 = (
#         "reports/charts/chart2.png"
#     )

#     fig = plt.figure(
#         figsize=(14,8)
#     )

#     ax1 = fig.add_subplot(
#         121
#     )

#     ax2 = fig.add_subplot(
#         122
#     )

#     ax1.imshow(
#         Image.open(chart1)
#     )

#     ax1.axis(
#         "off"
#     )

#     ax1.set_title(
#         "Chart 1"
#     )

#     ax2.imshow(
#         Image.open(chart2)
#     )

#     ax2.axis(
#         "off"
#     )

#     ax2.set_title(
#         "Chart 2"
#     )

#     plt.tight_layout()

#     plt.savefig(
#         dashboard
#     )

#     plt.close()

#     return dashboard




# import os
# from PIL import Image


# def create_dashboard(df):

#     os.makedirs(
#         "reports/dashboards",
#         exist_ok=True
#     )

#     chart1 = (
#         "reports/charts/chart1.png"
#     )

#     chart2 = (
#         "reports/charts/chart2.png"
#     )

#     dashboard = (
#         "reports/dashboards/dashboard.png"
#     )

#     img1 = Image.open(
#         chart1
#     )

#     img2 = Image.open(
#         chart2
#     )

#     width = img1.width + img2.width

#     height = max(
#         img1.height,
#         img2.height
#     )

#     final = Image.new(

#         "RGB",

#         (width, height),

#         "white"
#     )

#     final.paste(

#         img1,

#         (0, 0)
#     )

#     final.paste(

#         img2,

#         (
#             img1.width,
#             0
#         )
#     )

#     final.save(
#         dashboard
#     )

#     return dashboard




import os
import matplotlib.pyplot as plt


def create_dashboard(df):

    os.makedirs(
        "reports/dashboards",
        exist_ok=True
    )

    dashboard = (
        "reports/dashboards/dashboard.png"
    )

    numeric = df.select_dtypes(
        include="number"
    )

    fig = plt.figure(
        figsize=(18, 10)
    )

    # KPI

    plt.subplot(231)

    plt.axis("off")

    plt.text(
        0.2,
        0.5,
        f"Rows\n{len(df)}",
        fontsize=20
    )

    # Chart 1

    plt.subplot(232)

    numeric.iloc[:, 0].hist()

    plt.title(
        numeric.columns[0]
    )

    # Chart 2

    plt.subplot(233)

    numeric.iloc[:, 1].plot()

    plt.title(
        numeric.columns[1]
    )

    # Chart 3

    plt.subplot(234)

    numeric.mean().plot(
        kind="bar"
    )

    plt.title(
        "Average"
    )

    # Chart 4

    plt.subplot(235)

    numeric.sum().plot(
        kind="pie"
    )

    plt.title(
        "Contribution"
    )

    # Chart 5

    plt.subplot(236)

    numeric.boxplot()

    plt.title(
        "Distribution"
    )

    plt.tight_layout()

    plt.savefig(
        dashboard
    )

    plt.close()

    return dashboard