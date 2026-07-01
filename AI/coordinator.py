# # from agents.csv_agent import load_data
# # from agents.sql_agent import run_sql_analysis
# # from agents.chart_agent import create_charts
# # from agents.dashboard_agent import build_dashboard
# # from agents.insight_agent import generate_insights
# # from agents.report_agent import generate_report


# # def execute_pipeline(

# #     file,

# #     api

# # ):

# #     df = load_data(
# #         file
# #     )

# #     sql = run_sql_analysis(
# #         df
# #     )

# #     charts = create_charts(
# #         df
# #     )

# #     dashboard = build_dashboard()

# #     insights = generate_insights(
# #         df,
# #         api
# #     )

# #     report = generate_report(

# #         insights,

# #         dashboard
# #     )

# #     return {

# #         "df": df,

# #         "sql": sql,

# #         "dashboard": dashboard,

# #         "report": report,

# #         "insights": insights
# #     }




# from utils.data_loader import load_data

# from agents.sql_agent import run_sql
# from agents.chart_agent import create_charts
# from agents.dashboard_agent import create_dashboard
# from agents.insight_agent import generate_insights
# from agents.report_agent import generate_report


# def execute_pipeline(file):

#     # Load CSV
#     df = load_data(file)

#     # SQL Analysis
#     sql_result = run_sql(df)

#     # Charts
#     chart_paths = create_charts(df)

#     # Dashboard
#     dashboard_path = create_dashboard(df)

#     # Insights
#     insights = generate_insights(df)

#     # PDF Report
#     pdf_path = generate_report(
#         insights,
#         chart_paths,
#         dashboard_path
#     )

#     return {
#         "data": df,
#         "sql": sql_result,
#         "charts": chart_paths,
#         "dashboard": dashboard_path,
#         "insights": insights,
#         "report": pdf_path
#     }\\\



# from utils.data_loader import load_data

# from agents.sql_agent import run_sql

# from agents.chart_agent import create_charts

# from agents.dashboard_agent import create_dashboard

# from agents.insight_agent import generate_insights

# from agents.report_agent import generate_report


# def execute_pipeline(

#     file,

#     api
# ):

#     # ==========================
#     # LOAD DATA
#     # ==========================

#     df = load_data(file)

#     # ==========================
#     # SQL AGENT
#     # ==========================

#     sql = run_sql(df)

#     # ==========================
#     # CHART AGENT
#     # ==========================

#     charts = create_charts(df)

#     # ==========================
#     # ADD THIS HERE
#     # ==========================

#     dashboard = create_dashboard(df)

#     # ==========================
#     # INSIGHT AGENT
#     # ==========================

#     insights = generate_insights(

#         df,

#         api
#     )

#     # ==========================
#     # REPORT
#     # ==========================

#     report = generate_report(

#         insights,

#         dashboard
#     )

#     return {

#         "df": df,

#         "charts": charts,

#         "dashboard": dashboard,

#         "report": report,

#         "insights": insights
#     }




from utils.data_loader import load_data

from agents.sql_agent import run_sql

from agents.chart_agent import create_charts

from agents.dashboard_agent import create_dashboard

from agents.insight_agent import generate_insights

from agents.report_agent import generate_report


def execute_pipeline(

    file,

    api

):

    # LOAD CSV
    df = load_data(file)

    # SQL AGENT
    sql = run_sql(

        df,

        "describe"
    )

    # CHART AGENT
    create_charts(df)

    # DASHBOARD AGENT
    dashboard = create_dashboard(df)

    # INSIGHT AGENT
    insights = generate_insights(

        df,

        api
    )

    # REPORT AGENT
    report = generate_report(

        insights,

        dashboard
    )

    return {

        "df": df,

        "sql": sql,

        "dashboard": dashboard,

        "insights": insights,

        "report": report
    }