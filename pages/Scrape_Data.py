import streamlit as st
import pandas as pd
import re
from db import insert_product, fetch_all_products
from crawler import get_markdown
from langchain.embeddings import HuggingFaceEmbeddings

# Initialize the embeddings model
embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def parse_markdown(markdown_text: str):
    """
    Parse the markdown text to extract product details.
    Adjust the regex based on your actual markdown structure.
    """
    products = []
    # For demonstration, assume each product is separated by two newlines.
    entries = re.split(r'\n\s*\n', markdown_text.strip())
    for entry in entries:
        # Extract title from a markdown link: [Title](URL)
        title_match = re.search(r'\[([^\]]+)\]', entry)
        title = title_match.group(1) if title_match else "Unknown Title"
        # Extract price (assume format like ₹xx,xxx)
        price_match = re.search(r'₹[\d,]+', entry)
        price = price_match.group(0) if price_match else "₹0"
        # Extract rating (assume pattern: (x.x out of 5 stars))
        rating_match = re.search(r'\(([\d\.]+ out of 5 stars)\)', entry)
        rating = rating_match.group(1) if rating_match else "No rating"
        # Create product dict
        product = {
            "source": "markdown",
            "Laptop_Name": title,
            "Rating": rating,
            "Number_of_reviews": 0,  # Could be parsed if available
            "Discounted_price": int(price.replace("₹", "").replace(",", "")),
            "Original_price": 0,
            "Discount_percent": "N/A",
            "Benefits": "N/A",
            "Delivery_Date": "N/A",
            "Fast_Delivery": "N/A",
            "Sponsored_data": "N/A"
        }
        # Generate embedding from combined text fields
        text_to_embed = f"{title} {rating} {price}"
        product["embedding"] = embeddings_model.embed_query(text_to_embed)
        products.append(product)
    return products


def app():
    st.header("Live Scrape Data")
    st.info("This page scrapes live product data from Amazon.in.")

    with st.expander("How It Works"):
        st.write(
            "The crawler fetches product data in markdown format. The data is then parsed, enriched with embeddings, and stored in the local database (tagged as 'markdown').")

    if st.button("Scrape Data"):
        with st.spinner("Scraping data from Amazon.in..."):
            url = "https://www.amazon.in/s?k=laptops&crid=18QQ36HNW749J&sprefix=laptops%2Caps%2C602&ref=nb_sb_noss_2"
            try:
                markdown_data = get_markdown(url)
            except Exception as e:
                st.error(f"Error during scraping: {e}")
                markdown_data = ""
        if markdown_data:
            st.success("Scraping complete!")
            st.subheader("Raw Markdown Output")
            st.code(markdown_data, language="markdown")

            products = parse_markdown(markdown_data)
            st.subheader("Parsed Product Data Preview")
            if products:
                df = pd.DataFrame(products)
                st.data_editor(df, num_rows="dynamic", key="scraped_data")
                with st.spinner("Storing data into the database..."):
                    for prod in products:
                        insert_product(prod)
                    st.success("Data stored successfully!")
            else:
                st.warning("No products parsed from the markdown.")
        else:
            st.error("No markdown data received.")

    st.subheader("Stored Products (SQLite)")
    all_products = fetch_all_products("markdown")
    if all_products:
        df_db = pd.DataFrame(all_products, columns=[
            "ID", "Source", "Laptop_Name", "Rating", "Number_of_reviews", "Discounted_price",
            "Original_price", "Discount_percent", "Benefits", "Delivery_Date",
            "Fast_Delivery", "Sponsored_data", "Embedding"
        ])
        st.dataframe(df_db.drop(columns=["Embedding"]))
    else:
        st.info("No markdown-sourced products found in the database.")
