# from reportlab.pdfgen import canvas


# def generate_report(text):

#     pdf="report.pdf"

#     c=canvas.Canvas(pdf)

#     c.drawString(

#         50,

#         800,

#         text[:3000]

#     )

#     c.save()

#     return pdf





# import os

# from reportlab.platypus import (
#     SimpleDocTemplate,
#     Paragraph,
#     Spacer,
#     Image
# )

# from reportlab.lib.styles import (
#     getSampleStyleSheet
# )


# def generate_report(

#     insights,

#     dashboard_path

# ):

#     os.makedirs(

#         "reports/reports_pdf",

#         exist_ok=True
#     )

#     report_path = (

#         "reports/reports_pdf/report.pdf"
#     )

#     pdf = SimpleDocTemplate(

#         report_path
#     )

#     styles = getSampleStyleSheet()

#     content = []

#     # TITLE

#     content.append(

#         Paragraph(

#             "AI DATA ANALYST REPORT",

#             styles["Title"]
#         )
#     )

#     content.append(

#         Spacer(

#             1,

#             20
#         )
#     )

#     # INSIGHTS

#     content.append(

#         Paragraph(

#             insights,

#             styles["BodyText"]
#         )
#     )

#     content.append(

#         Spacer(

#             1,

#             30
#         )
#     )

#     # DASHBOARD IMAGE

#     if os.path.exists(

#         dashboard_path
#     ):

#         dashboard = Image(

#             dashboard_path,

#             width=500,

#             height=250
#         )

#         content.append(

#             dashboard
#         )

#     pdf.build(

#         content
#     )

#     return report_path




from reportlab.platypus import *

from reportlab.lib.styles import (
    getSampleStyleSheet
)

import os


def generate_report(

    insights,

    dashboard

):

    os.makedirs(

        "reports/reports_pdf",

        exist_ok=True
    )

    path = (

        "reports/reports_pdf/report.pdf"
    )

    doc = SimpleDocTemplate(
        path
    )

    styles = getSampleStyleSheet()

    data = []

    # Title

    data.append(

        Paragraph(

            "AI DATA ANALYST REPORT",

            styles["Title"]
        )
    )

    data.append(

        Spacer(

            1,

            20
        )
    )

    # Dashboard

    data.append(

        Paragraph(

            "Dashboard",

            styles["Heading1"]
        )
    )

    data.append(

        Image(

            dashboard,

            width=700,

            height=400
        )
    )

    data.append(

        Spacer(

            1,

            20
        )
    )

    # Insights

    data.append(

        Paragraph(

            "AI Insights",

            styles["Heading1"]
        )
    )

    data.append(

        Paragraph(

            insights,

            styles["BodyText"]
        )
    )

    doc.build(
        data
    )

    return path