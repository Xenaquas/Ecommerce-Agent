import streamlit as st
import pandas as pd
from db import fetch_all_products
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate
)
from langchain_ollama import ChatOllama

# Initialize the local LLM engine (Deepseek-R1)
llm_engine = ChatOllama(
    model="deepseek-r1:1.5b",
    base_url="http://localhost:11434",
    temperature=0.3,
    max_tokens=1000
)

# Custom CSS for the think process block
think_css = """
<style>
.think-block {
    background-color: #eef9ff;
    border-left: 5px solid #007acc;
    padding: 10px;
    margin: 10px 0;
    font-style: italic;
    font-family: monospace;
}
</style>
"""
st.markdown(think_css, unsafe_allow_html=True)


def build_prompt_chain(query, products, answer_detail, data_source):
    # System prompt instructs the model to output proper markdown without literal "\n" escapes.
    system_prompt = SystemMessagePromptTemplate.from_template(
        "You are an expert e-commerce data analyst. Provide a well-structured, professional markdown answer. "
        "Ensure your output uses actual line breaks and standard markdown table syntax, and do not output literal '\\n' characters."
    )

    # Build the product table in markdown format
    table_md = "| Laptop Name | Rating | Reviews | Price (₹) |\n"
    table_md += "|-------------|--------|---------|-----------|\n"
    for prod in products:
        table_md += f"| {prod['Laptop_Name']} | {prod['Rating']} | {prod['Number_of_reviews']} | {prod['Discounted_price']} |\n"

    detail_text = (
        "Provide a detailed analysis with insights, trends, and recommendations."
        if answer_detail else
        "Provide a concise summary highlighting key points."
    )

    human_prompt_text = (
        f"User Query: {query}\n\n"
        f"Data Source: {data_source}\n\n"
        f"Answer Style: {'Detailed' if answer_detail else 'Summarized'}\n\n"
        f"Product Data:\n{table_md}\n\n"
        "<think>\nBefore finalizing your answer, analyze the product data and extract key insights. "
        "Determine the best options within the specified price range.\n</think>\n\n"
        f"{detail_text}\n\n"
        "Please generate a structured markdown answer with proper line breaks and headers."
    )

    human_prompt = HumanMessagePromptTemplate.from_template(human_prompt_text)
    prompt_chain = ChatPromptTemplate.from_messages([system_prompt, human_prompt])
    return prompt_chain


def generate_llm_response(prompt_chain):
    processing_pipeline = prompt_chain | llm_engine
    response = processing_pipeline.invoke({})
    return response


def app():
    st.header("Query Data with Local LLM")
    st.info(
        "Enter your query about the product data. Choose the data source and select if you want a detailed or summarized answer. "
        "The query and relevant product data will be processed by the local LLM (Deepseek‑R1) to produce a structured markdown answer."
    )

    # Data source selection dropdown
    data_source = st.selectbox("Select Data Source", options=["markdown", "excel"], index=0)
    query_input = st.text_input("Your Query:")

    with st.expander("Optional Filters"):
        title_filter = st.text_input("Filter by Laptop Name (optional):")
        min_price = st.number_input("Min Price (₹)", value=0)
        max_price = st.number_input("Max Price (₹)", value=100000)

    answer_detail = st.checkbox("Provide Detailed Answer", value=False)

    if st.button("Run Query") and query_input:
        with st.spinner("Processing your query..."):
            all_products = fetch_all_products(data_source)
            products = []
            for prod in all_products:
                products.append({
                    "Laptop_Name": prod[2],
                    "Rating": prod[3],
                    "Number_of_reviews": prod[4],
                    "Discounted_price": prod[5],
                })

            def price_to_int(price):
                try:
                    return int(price)
                except:
                    return 0

            filtered = []
            for prod in products:
                if title_filter and title_filter.lower() not in prod["Laptop_Name"].lower():
                    continue
                if price_to_int(prod["Discounted_price"]) < min_price or price_to_int(
                        prod["Discounted_price"]) > max_price:
                    continue
                filtered.append(prod)
            if not filtered:
                filtered = products

            prompt_chain = build_prompt_chain(query_input, filtered, answer_detail, data_source)
            llm_response = generate_llm_response(prompt_chain)
            # Use the content of the AIMessage and fix newline issues
            formatted_response = llm_response.content.replace("\\n", "\n")

        st.markdown("## Structured Query Answer")
        st.markdown(formatted_response)

        st.subheader("Price Distribution of Matching Products")
        try:
            import plotly.express as px
            df = pd.DataFrame(filtered)
            df["PriceInt"] = df["Discounted_price"].apply(lambda x: int(x))
            fig = px.histogram(df, x="Price", nbins=10, title="Price Distribution")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error generating chart: {e}")
