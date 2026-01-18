import streamlit as st
import pandas as pd
from dualrecommender import DualRecommender

st.set_page_config(
    page_title="Pharmacy Item Recommendation",
    layout="centered"
)

st.title("K-24 Item Recommendation System")
st.caption("🛒 = Pola beli (IBCF) | 🧬 = Kemiripan kandungan, golongan, kategori, dan satuan obat")

@st.cache_data
def load_items():
    return pd.read_csv("data/items_dummy.csv")

items = load_items()

# Load model
@st.cache_resource(show_spinner="Loading recommender system...")
def load_model(items):
    model = DualRecommender(
        items=items,
        enable_content=True
    )
    model.load_ibcf("artifacts/ibcf_sim.parquet")
    return model

model = load_model(items)

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

# Recommendation
if st.button("Search"):
    result = model.recommend(selected_item)

    # IBCF
    if not result["ibcf"].empty:
        st.subheader("🛒 Sering Dibeli Bersama")
        for _, row in result["ibcf"].iterrows():
            st.markdown(
                f"**{row['Nama_Item']}**  \n"
                f"Kandungan: *{row['Kandungan']}*  \n"
                f"Skor relevansi: `{row['Score']:.2f}`"
            )
    else:
        st.info("Tidak cukup data pola pembelian untuk item ini.")

    st.divider()

    # Content
    if not result["content"].empty:
        st.subheader("🧬 Mirip Secara Medis")
        for _, row in result["content"].iterrows():
            st.markdown(
                f"**{row['Nama_Item']}**  \n"
                f"Kandungan: *{row['Kandungan']}*  \n"
                f"Skor kemiripan: `{row['Score']:.2f}`"
            )
    else:
        st.info("Tidak ada item dengan kemiripan konten yang cukup.")
