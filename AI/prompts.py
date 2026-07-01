# utils/prompts.py


INSIGHT_PROMPT = """

You are an expert AI Data Analyst.

Dataset Information:

{data}

Generate:

1. Dataset Summary

2. Top Insights

3. Trends

4. Recommendations

5. Business Decisions

Keep explanation simple.

"""


REPORT_PROMPT = """

Generate Final Report.

Include:

Dataset Summary

Insights

Charts Explanation

Recommendations

Conclusion

"""


SQL_PROMPT = """

Analyze dataset.

Generate:

Row Count

Column Count

Missing Values

Numeric Statistics

Most Important Columns

"""