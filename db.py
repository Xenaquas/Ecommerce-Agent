import sqlite3
import faiss
import numpy as np
import pickle

DB_PATH = "ecommerce.db"
FAISS_INDEX_PATH = "faiss.index"
EMBEDDINGS_PATH = "embeddings.pkl"  # Optional: store embeddings list

def init_sqlite_db():
    """Initialize the SQLite database and create the products table."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT,  -- "markdown" or "excel"
            Laptop_Name TEXT,
            Rating TEXT,
            Number_of_reviews INTEGER,
            Discounted_price INTEGER,
            Original_price INTEGER,
            Discount_percent TEXT,
            Benefits TEXT,
            Delivery_Date TEXT,
            Fast_Delivery TEXT,
            Sponsored_data TEXT,
            embedding BLOB
        )
    ''')
    conn.commit()
    conn.close()

def insert_product(product):
    """
    Insert a product record into the database.
    The product dict should contain a key "source" indicating the data source ("markdown" or "excel").
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    emb_blob = pickle.dumps(product.get("embedding"))
    c.execute('''
        INSERT INTO products (
            source, Laptop_Name, Rating, Number_of_reviews, Discounted_price,
            Original_price, Discount_percent, Benefits, Delivery_Date,
            Fast_Delivery, Sponsored_data, embedding
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        product.get("source"),
        product.get("Laptop_Name"),
        product.get("Rating"),
        product.get("Number_of_reviews"),
        product.get("Discounted_price"),
        product.get("Original_price"),
        product.get("Discount_percent"),
        product.get("Benefits"),
        product.get("Delivery_Date"),
        product.get("Fast_Delivery"),
        product.get("Sponsored_data"),
        emb_blob
    ))
    conn.commit()
    conn.close()

def fetch_all_products(source_filter=None):
    """Retrieve all product records, optionally filtering by source."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    if source_filter:
        c.execute("SELECT * FROM products WHERE source = ?", (source_filter,))
    else:
        c.execute("SELECT * FROM products")
    rows = c.fetchall()
    conn.close()
    return rows

def build_faiss_index(dimension=768):
    """
    Build a Faiss index from product embeddings stored in the database.
    """
    products = fetch_all_products()
    embeddings = []
    for prod in products:
        emb = pickle.loads(prod[-1])
        embeddings.append(emb)
    if not embeddings:
        return None
    vectors = np.array(embeddings).astype("float32")
    index = faiss.IndexFlatL2(dimension)
    index.add(vectors)
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(EMBEDDINGS_PATH, "wb") as f:
        pickle.dump(embeddings, f)
    return index

# Initialize the database on module import
init_sqlite_db()
