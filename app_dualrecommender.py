import streamlit as st
import pandas as pd
from dualrecommender import DualRecommender

st.set_page_config(page_title="Pharmacy Item Recommendation", layout="centered")

st.title("K-24 Item Recommendation System")
st.caption("🛒 = Pola beli (co occurrence)| 🧬 = Kemiripan kandungan, golongan, dan satuan obat")

@st.cache_data
def load_data():
    transactions = pd.read_csv("data/transactions_dummy.csv")
    details = pd.read_csv("data/transaction_details_dummy.csv")
    items = pd.read_csv("data/items_dummy.csv")
    return transactions, details, items

transactions, details, items = load_data()

# @st.cache_resource
def load_model(details, items, transactions):
    return DualRecommender(details, items, transactions)

model = load_model(details, items, transactions)

item_list = {
    row["Kode_Item"]: (
        f"{row['Nama_Item']} — {row['Kandungan']} "
        f"({row['Kategori_Obat']})"
    )
    for _, row in items.iterrows()
}

selected_item = st.selectbox(
    "Pilih obat yang dibeli:",
    options=list(item_list.keys()),
    format_func=lambda x: item_list[x]
)

if st.button("Search"):
    result = model.recommend(selected_item)

    if not result["co_occurrence"].empty:
        st.subheader("🛒 Sering Dibeli Bersama")
        for _, row in result["co_occurrence"].iterrows():
            st.markdown(
                f"**{row['Nama_Item']}**  \n"
                f"Kandungan: *{row['Kandungan']}*  \n"
                f"Skor relevansi: `{row['Score']:.2f}`"
            )
    else:
        st.info("Tidak cukup data co-occurrence untuk item ini.")

    st.divider()

    if not result["content_based"].empty:
        st.subheader("🧬 Mirip Secara Medis")
        for _, row in result["content_based"].iterrows():
            st.markdown(
                f"**{row['Nama_Item']}**  \n"
                f"Kandungan: *{row['Kandungan']}*  \n"
                f"Skor kemiripan: `{row['Score']:.2f}`"
            )
    else:
        st.info("Tidak ada item dengan kemiripan konten yang cukup.")
