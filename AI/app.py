# import streamlit as st
# import pandas as pd
# import os

# from agents.coordinator import execute_pipeline


# # =====================================
# # PAGE CONFIG
# # =====================================

# st.set_page_config(

#     page_title="AI Data Analyst",

#     page_icon="📊",

#     layout="wide"
# )


# # =====================================
# # TITLE
# # =====================================

# st.title(
#     "📊 AI Data Analyst (Multi-Agent)"
# )


# # =====================================
# # SIDEBAR
# # =====================================

# st.sidebar.title(
#     "⚙ Settings"
# )

# groq_api = st.sidebar.text_input(

#     "Enter GROQ API Key",

#     type="password"
# )


# # =====================================
# # SESSION
# # =====================================

# if "result" not in st.session_state:

#     st.session_state.result = None


# # =====================================
# # UPLOAD
# # =====================================

# uploaded_file = st.file_uploader(

#     "📂 Upload CSV",

#     type=["csv"]
# )


# # =====================================
# # ANALYZE
# # =====================================

# if st.button(
#     "🚀 Analyze Dataset"
# ):

#     if not uploaded_file:

#         st.error(
#             "Upload CSV"
#         )

#         st.stop()

#     if not groq_api:

#         st.error(
#             "Enter API Key"
#         )

#         st.stop()

#     with st.spinner(

#         "AI Agents Working..."
#     ):

#         result = execute_pipeline(

#             uploaded_file,

#             groq_api
#         )

#         st.session_state.result = result


# # =====================================
# # SHOW RESULTS
# # =====================================

# if st.session_state.result:

#     result = st.session_state.result

#     st.success(
#         "Analysis Completed"
#     )

#     st.markdown("---")


#     # ==========================
#     # DATA
#     # ==========================

#     st.header(
#         "📄 Dataset Preview"
#     )

#     st.dataframe(

#         result["df"],

#         use_container_width=True
#     )


#     st.markdown("---")


#     # ==========================
#     # DASHBOARD
#     # ==========================

#     st.header(
#         "📈 Dashboard"
#     )

#     if os.path.exists(

#         result["dashboard"]

#     ):

#         st.image(

#             result["dashboard"],

#             use_container_width=True
#         )


#     st.markdown("---")


#     # ==========================
#     # CHARTS
#     # ==========================

#     st.header(
#         "📊 Charts"
#     )

#     charts = [

#         "reports/charts/chart1.png",

#         "reports/charts/chart2.png"
#     ]

#     cols = st.columns(2)

#     for i, chart in enumerate(charts):

#         if os.path.exists(chart):

#             with cols[i]:

#                 st.image(

#                     chart,

#                     use_container_width=True
#                 )


#     st.markdown("---")


#     # ==========================
#     # INSIGHTS
#     # ==========================

#     st.header(
#         "🧠 AI Insights"
#     )

#     st.write(

#         result["insights"]
#     )


#     st.markdown("---")


#     # ==========================
#     # REPORT
#     # ==========================

#     st.header(
#         "📑 Download Report"
#     )

#     if os.path.exists(

#         result["report"]

#     ):

#         with open(

#             result["report"],

#             "rb"

#         ) as pdf:

#             st.download_button(

#                 label="⬇ Download PDF Report",

#                 data=pdf,

#                 file_name="AI_Report.pdf",

#                 mime="application/pdf"
#             )


#     st.markdown("---")


#     # ==========================
#     # SAVE LOCATION
#     # ==========================

#     st.success(

#         """
# Reports Saved Successfully

# charts → reports/charts

# dashboard → reports/dashboards

# pdf → reports/reports_pdf
# """
#     )


# # =====================================
# # FOOTER
# # =====================================

# st.markdown("---")

# st.caption(
#     "AI Data Analyst • Multi-Agent • Dashboard • Reports"
# )





import streamlit as st
import pandas as pd
from agents.coordinator import execute_pipeline


st.set_page_config(

    page_title="AI Data Analyst",

    page_icon="📊",

    layout="wide"
)

st.title(
    "📊 AI Data Analyst (Multi-Agent System)"
)

# ===============================
# SIDEBAR
# ===============================

st.sidebar.header(
    "Settings"
)

groq_api = st.sidebar.text_input(

    "Groq API Key",

    type="password"
)

# ===============================
# FILE UPLOAD
# ===============================

uploaded = st.file_uploader(

    "Upload CSV",

    type=["csv"]
)

# ===============================
# RUN
# ===============================

if uploaded:

    df = pd.read_csv(
        uploaded
    )

    st.subheader(
        "Dataset Preview"
    )

    st.dataframe(
        df.head()
    )

    if st.button(
        "Generate AI Dashboard"
    ):

        if not groq_api:

            st.error(
                "Enter API Key"
            )

            st.stop()

        with st.spinner(

            "Running Agents..."
        ):

            result = execute_pipeline(

                uploaded,

                groq_api
            )

        st.success(
            "Analysis Completed"
        )

        # ==========================
        # DASHBOARD
        # ==========================

        st.markdown("---")

        st.header(
            "📈 Executive Dashboard"
        )

        st.image(

            result["dashboard"],

            use_container_width=True
        )

        # ==========================
        # INSIGHTS
        # ==========================

        st.markdown("---")

        st.header(
            "🧠 AI Insights"
        )

        st.write(

            result["insights"]
        )

        # ==========================
        # SQL OUTPUT
        # ==========================

        st.markdown("---")

        st.header(
            "🗃 SQL Agent"
        )

        st.write(

            result["sql"]
        )

        # ==========================
        # REPORT DOWNLOAD
        # ==========================

        st.markdown("---")

        st.header(
            "📄 Download Report"
        )

        with open(

            result["report"],

            "rb"

        ) as f:

            st.download_button(

                label="⬇ Download PDF Report",

                data=f,

                file_name="AI_Report.pdf",

                mime="application/pdf"
            )

        # ==========================
        # GENERATED FILES
        # ==========================

        st.markdown("---")

        st.header(
            "📁 Generated Assets"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.image(

                "reports/charts/chart1.png"
            )

        with col2:

            st.image(

                "reports/charts/chart2.png"
            )

else:

    st.info(
        "Upload CSV to start"
    )