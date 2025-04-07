import streamlit as st
import pandas as pd
from db import insert_product, fetch_all_products
from langchain.embeddings import HuggingFaceEmbeddings

embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def load_excel_data():
    """Load product data from the Excel file."""
    try:
        df = pd.read_excel("data.xlsx")
        return df
    except Exception as e:
        st.error(f"Error loading Excel file: {e}")
        return pd.DataFrame()


def store_products_from_df(df: pd.DataFrame):
    """Store each product from the DataFrame into the database with source 'excel'."""
    for _, row in df.iterrows():
        text_to_embed = f"{row['Laptop_Name']} {row['Rating']} ₹{row['Discounted_price']}"
        embedding = embeddings_model.embed_query(text_to_embed)
        product = {
            "source": "excel",
            "Laptop_Name": row.get("Laptop_Name"),
            "Rating": row.get("Rating"),
            "Number_of_reviews": int(row.get("Number_of_reviews", 0)),
            "Discounted_price": int(row.get("Discounted_price", 0)),
            "Original_price": int(row.get("Original_price", 0)),
            "Discount_percent": row.get("Discount_percent"),
            "Benefits": row.get("Benefits"),
            "Delivery_Date": row.get("Delivery_Date"),
            "Fast_Delivery": row.get("Fast_Delivery"),
            "Sponsored_data": row.get("Sponsored_data"),
            "embedding": embedding
        }
        insert_product(product)


def app():
    st.header("Excel Data Demo")
    st.info("Load product data from an Excel file (demo fallback).")

    uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])
    if uploaded_file:
        df = pd.read_excel(uploaded_file)
    else:
        df = load_excel_data()

    if not df.empty:
        st.subheader("Excel Data Preview")
        st.dataframe(df)
        if st.button("Store Excel Data in Database"):
            with st.spinner("Storing Excel data..."):
                store_products_from_df(df)
            st.success("Excel data stored successfully!")
    else:
        st.warning("No Excel data available.")

    st.subheader("Stored Products (SQLite) from Excel")
    all_products = fetch_all_products("excel")
    if all_products:
        df_db = pd.DataFrame(all_products, columns=[
            "ID", "Source", "Laptop_Name", "Rating", "Number_of_reviews", "Discounted_price",
            "Original_price", "Discount_percent", "Benefits", "Delivery_Date",
            "Fast_Delivery", "Sponsored_data", "Embedding"
        ])
        st.dataframe(df_db.drop(columns=["Embedding"]))
    else:
        st.info("No excel-sourced products found in the database.")
