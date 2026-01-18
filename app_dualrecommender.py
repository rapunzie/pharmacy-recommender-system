import streamlit as st
import pandas as pd
from dualrecommender import DualRecommender

st.set_page_config(page_title="Pharmacy Item Recommendation", layout="centered")

st.title("K-24 Item Recommendation System")
st.caption("🛒 = Pola beli (co-occurrence) | 🧬 = Kemiripan kandungan, golongan, satuan obat")

@st.cache_data
def load_data():
    return (
        pd.read_csv("data/transactions_dummy.csv"),
        pd.read_csv("data/transaction_details_dummy.csv"),
        pd.read_csv("data/items_dummy.csv")
    )

transactions, details, items = load_data()

@st.cache_resource
def load_model(details, items, transactions):
    return DualRecommender(details, items, transactions)

model = load_model(details, items, transactions)

item_list = {
    row["Kode_Item"]: f"{row['Nama_Item']} — {row['Kandungan']} ({row['Kategori_Obat']})"
    for _, row in items.iterrows()
}

selected_item = st.selectbox("Pilih obat:", list(item_list.keys()),
                             format_func=lambda x: item_list[x])

if st.button("Search"):
    result = model.recommend(selected_item)

    st.subheader("🛒 Sering Dibeli Bersama")
    if result["co_occurrence"].empty:
        st.info("Tidak cukup data.")
    else:
        for _, r in result["co_occurrence"].iterrows():
            st.write(f"**{r['Nama_Item']}** — {r['Kandungan']} ({r['Score']:.2f})")

    st.subheader("🧬 Mirip Secara Medis")
    if result["content_based"].empty:
        st.info("Tidak ada item mirip.")
    else:
        for _, r in result["content_based"].iterrows():
            st.write(f"**{r['Nama_Item']}** — {r['Kandungan']} ({r['Score']:.2f})")
