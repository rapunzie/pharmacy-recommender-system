import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


class DualRecommender:
    def __init__(
        self,
        items: pd.DataFrame,
        enable_content: bool = True,
        max_tfidf_features: int = None
    ):
        self.items = items.copy()
        self.enable_content = enable_content
        self.max_tfidf_features = max_tfidf_features

        # Lookups
        self.item_lookup = (
            self.items.set_index("Kode_Item")["Nama_Item"].to_dict()
        )
        self.kandungan_lookup = (
            self.items.set_index("Kode_Item")["Kandungan"].to_dict()
        )

        # Models
        self.ibcf_sim = None
        self.content_sim = None

        if self.enable_content:
            self._build_content()

    #IBCF Pre compute
    def load_ibcf(self, path: str):
        self.ibcf_sim = pd.read_parquet(path)

    #Content based
    def _build_content(self):
        self.items["content"] = (
            self.items["Kandungan"].fillna("") + " " +
            self.items["Kategori_Obat"].fillna("") + " " +
            self.items["Tipe_Golongan_Obat"].fillna("") + " " +
            self.items["Satuan"].fillna("")
        )

        tfidf = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=5  
        )

        mat = tfidf.fit_transform(self.items["content"])
        sim = cosine_similarity(mat)

        self.content_sim = pd.DataFrame(
            sim,
            index=self.items["Kode_Item"],
            columns=self.items["Kode_Item"]
        )

    @staticmethod
    def _normalize(s: pd.Series):
        return (s - s.min()) / (s.max() - s.min() + 1e-9)

    def recommend(self, item_code: str, top_k: int = 5):
        results = {}

        # IBCF
        if self.ibcf_sim is not None and item_code in self.ibcf_sim.columns:
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
            co["Method"] = "IBCF"

            results["ibcf"] = co
        else:
            results["ibcf"] = pd.DataFrame()

        # Content
        if self.content_sim is not None and item_code in self.content_sim.columns:
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

            results["content"] = cb
        else:
            results["content"] = pd.DataFrame()

        return results
