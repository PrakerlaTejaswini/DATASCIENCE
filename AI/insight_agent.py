# from langchain_groq import ChatGroq


# def generate_insight(api_key,data):

#     llm=ChatGroq(

#         groq_api_key=api_key,

#         model_name="llama-3.3-70b-versatile"

#     )

#     prompt=f"""

# Analyze:

# {data}

# Generate:

# 1 Summary

# 2 Insights

# 3 Recommendations

# """

#     result=llm.invoke(prompt)

#     return result.content




from langchain_groq import ChatGroq


def generate_insights(

    df,

    api_key

):

    llm = ChatGroq(

        groq_api_key=api_key,

        model_name="llama-3.3-70b-versatile",

        temperature=0
    )

    dataset_info = f"""

Rows:
{len(df)}

Columns:
{list(df.columns)}

Statistics:
{df.describe(include="all").to_string()}

"""

    prompt = f"""

You are a Senior AI Data Analyst.

Analyze the dataset.

Provide:

1. Dataset Summary
2. Key Trends
3. Missing Values Analysis
4. Business Insights
5. Recommendations

Dataset:

{dataset_info}

"""

    response = llm.invoke(

        prompt
    )

    return response.content