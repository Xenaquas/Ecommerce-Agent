# Fast Ecommerce Data Agent

Fast Ecommerce Data Agent is a Streamlit-based web application designed to efficiently process and analyze e-commerce product data. The application leverages local LLM inference with Deepseek-R1, using advanced optimizations such as bitsandbytes quantization, GPU acceleration (via CUDA), and ONNX Runtime for faster inference. Data is loaded from an Excel file (`data.xlsx`), filtered based on user queries, and the filtered context is passed to the LLM to generate a structured markdown answer.

## About

This project was built as an Agentic AI tool to help users quickly extract insights from e-commerce product data. It is optimized for speed by:
- Using a smaller or quantized model with bitsandbytes to reduce memory footprint.
- Ensuring GPU usage through CUDA if available.
- Optionally exporting the model to ONNX and using ONNX Runtime for even faster inference.
- Caching data and model loads using Streamlit’s caching decorators.

The app also includes interactive data visualizations (using Plotly) and basic chat memory for context retention.

## Features

- **Fast Inference:** Leverages GPU acceleration and model quantization.
- **ONNX Runtime Support:** Optional faster inference using an exported ONNX model.
- **Data Filtering:** Loads product data from an Excel file, applies filters, and converts filtered data into JSON context for the LLM.
- **Structured Responses:** Generates structured markdown answers, including a custom `<think>` block that outlines the reasoning process.
- **Interactive Visualizations:** Displays filtered product data and price distribution charts.
- **Result & Resource Caching:** Uses Streamlit caching to optimize performance.

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/Xenaquas/Ecommerce-Agent.git
   cd fast-ecommerce-data-agent
   
2. **Create a Virtual Environment (optional but recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate

3. **Install Requirements:**

    ```bash
    pip install -r requirements.txt
   
4. **Place Your Data File:**

Ensure that you have an Excel file named data.xlsx in the root directory. This file should contain the product data.

Usage
Run the App:

```bash
streamlit run app.py
````  
  
**Navigate the App:**

Enter Your Query: Input a query like "List all laptops under 50000 rupees" in the query box.

Apply Data Filters: Set minimum and maximum price ranges.

Run Query: Click the "Run Query" button to filter data, visualize product information, and generate a structured response.

View Performance Tips: The app displays performance recommendations and chat history for debugging.

Exporting the Model to ONNX (Optional)
To further speed up inference, you can export your model to ONNX format and use ONNX Runtime. For example:

```bash
python -m transformers.onnx --model deepseek-r1:1.5b --feature text-generation onnx/
```

Ensure the exported ONNX model is located at onnx/model.onnx (or update the path in the code accordingly).

Requirements
See requirements.txt for the complete list of dependencies.

Contributing
Contributions are welcome! Feel free to open issues or submit pull requests with improvements and new features.

License
This project is licensed under the MIT License.

**requirements.txt**

```txt
streamlit>=1.15.0
pandas>=1.3.0
torch>=1.10.0
transformers>=4.20.0
langchain>=0.0.148
langchain_ollama>=0.0.1  # or the specific version you are using
bitsandbytes>=0.39.0
onnxruntime>=1.10.0
plotly>=5.0.0
