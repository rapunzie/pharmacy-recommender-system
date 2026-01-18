import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


class DualRecommender:
    def __init__(
        self,
        details: pd.DataFrame,
        items: pd.DataFrame,
        transactions: pd.DataFrame,
        min_freq: int = 30,
        max_tfidf_features: int = 5000
    ):
        self.details = details.copy()
        self.items = items.copy()
        self.transactions = transactions.copy()
        self.min_freq = min_freq
        self.max_tfidf_features = max_tfidf_features

        self._prepare_data()

        # Lazy-loaded attributes
        self.ibcf_sim = None
        self.content_sim = None

    # Data preparation
    def _prepare_data(self):
        self.df = (
            self.details
            .merge(self.items, on="Kode_Item", how="left")
            .merge(
                self.transactions[["Kode_Transaksi", "Tanggal"]],
                on="Kode_Transaksi",
                how="left"
            )
        )

        self.df["Tanggal"] = pd.to_datetime(self.df["Tanggal"], errors="coerce")

        self.item_lookup = (
            self.items.set_index("Kode_Item")["Nama_Item"].to_dict()
        )
        self.kandungan_lookup = (
            self.items.set_index("Kode_Item")["Kandungan"].to_dict()
        )

    # Item-based Collaborative Filtering (Co-occurrence)
    def _build_ibcf(self):
        basket = (
            self.df
            .groupby(["Kode_Transaksi", "Kode_Item"])["qty"]
            .sum()
            .unstack(fill_value=0)
        ).clip(upper=5)

        item_freq = (basket > 0).sum(axis=0)
        frequent_items = item_freq[item_freq >= self.min_freq].index

        if len(frequent_items) == 0:
            self.ibcf_sim = pd.DataFrame()
            return

        basket = basket[frequent_items]
        sim_matrix = cosine_similarity(basket.T)

        self.ibcf_sim = pd.DataFrame(
            sim_matrix,
            index=basket.columns,
            columns=basket.columns
        )

    # Content-based Filtering
    def _build_content(self):
        self.items["content"] = (
            self.items["Kandungan"].fillna("") + " " +
            self.items["Kategori_Obat"].fillna("") + " " +
            self.items["Tipe_Golongan_Obat"].fillna("") + " " +
            self.items["Satuan"].fillna("")
        )

        tfidf = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_features=self.max_tfidf_features
        )

        tfidf_matrix = tfidf.fit_transform(self.items["content"])
        sim_matrix = cosine_similarity(tfidf_matrix)

        self.content_sim = pd.DataFrame(
            sim_matrix,
            index=self.items["Kode_Item"],
            columns=self.items["Kode_Item"]
        )

    @staticmethod
    def _normalize(series: pd.Series) -> pd.Series:
        return (series - series.min()) / (series.max() - series.min() + 1e-9)
  
    def recommend(self, item_code: str, top_k: int = 5):
  
        if self.ibcf_sim is None:
            self._build_ibcf()

        # if self.content_sim is None:
        #     self._build_content()

        results = {}

        # Co-occurance
        if item_code in self.ibcf_sim.columns:
            co = (
                self._normalize(self.ibcf_sim[item_code])
                .drop(item_code, errors="ignore")
                .sort_values(ascending=False)
                .head(top_k)
                .reset_index()
                .rename(columns={"index": "Kode_Item", item_code: "Score"})
            )

            co["Nama_Item"] = co["Kode_Item"].map(self.item_lookup)
            co["Kandungan"] = co["Kode_Item"].map(self.kandungan_lookup)
            co["Method"] = "Co-Occurrence"

            results["co_occurrence"] = co
        else:
            results["co_occurrence"] = pd.DataFrame()

        # Content-based
        if item_code in self.content_sim.columns:
            cb = (
                self._normalize(self.content_sim[item_code])
                .drop(item_code, errors="ignore")
                .sort_values(ascending=False)
                .head(top_k)
                .reset_index()
                .rename(columns={"index": "Kode_Item", item_code: "Score"})
            )

            cb["Nama_Item"] = cb["Kode_Item"].map(self.item_lookup)
            cb["Kandungan"] = cb["Kode_Item"].map(self.kandungan_lookup)
            cb["Method"] = "Content-Based"

            results["content_based"] = cb
        else:
            results["content_based"] = pd.DataFrame()

        return results
