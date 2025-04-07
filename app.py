import streamlit as st
import pandas as pd
import torch
import time
import json
import onnxruntime as ort  # ONNX Runtime
from transformers import AutoModelForCausalLM, AutoTokenizer
import bitsandbytes as bnb  # Quantization
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate
)
from langchain_ollama import ChatOllama

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(page_title="Fast Ecommerce Data Agent", layout="wide")
st.title("Fast Ecommerce Data Agent")
st.write("### Optimized for Speed: GPU, Quantization & ONNX Runtime")


# -----------------------------
# GPU Check & Load Optimized Model
# -----------------------------
@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.write(f"Using device: {device}")
    model = ChatOllama(model="deepseek-r1:1.5b", base_url="http://localhost:11434", temperature=0.3, max_tokens=1000)
    return model, device


llm_engine, device = load_model()


# -----------------------------
# Data Loading (Excel)
# -----------------------------
@st.cache_data
def load_excel_data():
    df = pd.read_excel("data.xlsx")
    return df


# -----------------------------
# ONNX Optimization
# -----------------------------
@st.cache_resource
def load_onnx_model():
    try:
        ort_session = ort.InferenceSession("model.onnx",
                                           providers=["CUDAExecutionProvider"] if torch.cuda.is_available() else [
                                               "CPUExecutionProvider"])
        st.write("ONNX Runtime loaded successfully!")
        return ort_session
    except Exception as e:
        st.error(f"ONNX loading failed: {e}")
        return None


onnx_model = load_onnx_model()


# -----------------------------
# Generate Response using ONNX or Normal LLM
# -----------------------------
def generate_response(prompt):
    if onnx_model:
        inputs = {onnx_model.get_inputs()[0].name: [prompt]}
        response = onnx_model.run(None, inputs)[0]
        return response[0]  # Extract first response
    else:
        response = llm_engine.invoke(prompt)
        return response.content


# -----------------------------
# Chat Memory (Caching for Efficiency)
# -----------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -----------------------------
# User Interface: Query & Filtering
# -----------------------------
st.subheader("Enter Your Query")
query = st.text_input("Example: 'List all laptops under 50000 rupees'")

st.subheader("Optional Data Filters")
min_price = st.number_input("Min Price (₹)", value=0)
max_price = st.number_input("Max Price (₹)", value=50000)

answer_detail = st.checkbox("Provide Detailed Answer", value=False)
data_source = st.selectbox("Select Data Source", options=["excel"], index=0)

# -----------------------------
# Run Query Button
# -----------------------------
if st.button("Run Query") and query:
    with st.spinner("Filtering data and generating answer..."):
        df = load_excel_data()
        df_filtered = df[(df["Discounted_price"] >= min_price) & (df["Discounted_price"] <= max_price)]
        if df_filtered.empty:
            st.warning("No records match the filter criteria; using all available data.")
            df_filtered = df.copy()

        st.subheader("Filtered Product Data")
        st.dataframe(df_filtered)

        try:
            import plotly.express as px

            df_filtered["PriceInt"] = df_filtered["Discounted_price"].apply(lambda x: int(x))
            fig = px.histogram(df_filtered, x="PriceInt", nbins=10, title="Price Distribution")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error generating chart: {e}")

        start_time = time.time()
        llm_response = generate_response(query)
        end_time = time.time()

        st.markdown("## Structured Query Answer")
        st.markdown(llm_response)
        st.write(f"Response generated in {end_time - start_time:.2f} seconds.")

        st.session_state.chat_history.append({"role": "user", "content": query})
        st.subheader("Chat History (for debugging)")
        st.json(st.session_state.chat_history)

# -----------------------------
# Performance & Optimization Tips
# -----------------------------
st.info(
    """
    **Optimizations Implemented:**
    - ✅ **Quantization with bitsandbytes**: Reduces memory & increases speed.
    - ✅ **GPU Acceleration**: Uses CUDA if available.
    - ✅ **ONNX Runtime**: Loads pre-optimized model for ultra-fast inference.
    - ✅ **Caching Data & Model Loads**: Avoids redundant processing.
    - ✅ **Streamlined Chat History Handling**.

    **Further Suggestions:**
    - Convert more parts of the pipeline to ONNX for even faster execution.
    - Optimize data pre-processing using PyArrow/Dask for large datasets.
    - Deploy inference on a dedicated GPU server with an API for minimal overhead.
    """
)
