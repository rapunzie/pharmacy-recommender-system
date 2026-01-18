import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

class DualRecommender:
    def __init__(self, details, items, transactions, min_freq=100):
        self.details = details
        self.items = items
        self.transactions = transactions
        self.min_freq = min_freq

        self._prepare()
        self._build_ibcf()
        self._build_content()

    def _prepare(self):
        self.df = (
            self.details
            .merge(self.items, on="Kode_Item", how="left")
            .merge(self.transactions[["Kode_Transaksi","Tanggal"]], 
                   on="Kode_Transaksi", how="left")
        )

        self.df["Tanggal"] = pd.to_datetime(self.df["Tanggal"])

        self.item_lookup = self.items.set_index("Kode_Item")["Nama_Item"].to_dict()
        self.kandungan_lookup = self.items.set_index("Kode_Item")["Kandungan"].to_dict()

    # Co occurance
    def _build_ibcf(self):
        basket = (
            self.df.groupby(["Kode_Transaksi", "Kode_Item"])["qty"]
            .sum()
            .unstack(fill_value=0)
        ).clip(upper=5)

        item_freq = (basket > 0).sum(axis=0)
        frequent_items = item_freq[item_freq >= self.min_freq].index

        basket = basket[frequent_items]
        sim = cosine_similarity(basket.T)

        self.ibcf_sim = pd.DataFrame(
            sim, index=basket.columns, columns=basket.columns
        )

    # Content based
    def _build_content(self):
        self.items["content"] = (
            self.items["Kandungan"].fillna("") + " " +
            self.items["Kategori_Obat"].fillna("") + " " +
            self.items["Tipe_Golongan_Obat"].fillna("") + " " +
            self.items["Satuan"].fillna("")
        )

        tfidf = TfidfVectorizer(ngram_range=(1,2), min_df=5)
        mat = tfidf.fit_transform(self.items["content"])

        sim = cosine_similarity(mat)
        self.content_sim = pd.DataFrame(
            sim,
            index=self.items["Kode_Item"],
            columns=self.items["Kode_Item"]
        )

    # Recommend
    def _normalize(self, s):
        return (s - s.min()) / (s.max() - s.min() + 1e-9)

    def recommend(self, item_code, top_k=5):

        # Co-occurance
        if item_code in self.ibcf_sim:
            co = (
                self._normalize(self.ibcf_sim[item_code])
                .drop(item_code, errors="ignore")
                .sort_values(ascending=False)
                .head(top_k)
                .reset_index()
                .rename(columns={"index":"Kode_Item", item_code:"Score"})
                .assign(
                    Nama_Item=lambda x: x["Kode_Item"].map(self.item_lookup),
                    Kandungan=lambda x: x["Kode_Item"].map(self.kandungan_lookup),
                    Method="Co-Occurrence"
                )
            )
        else:
            co = pd.DataFrame()

        # Content-based
        if item_code in self.content_sim:
            cb = (
                self._normalize(self.content_sim[item_code])
                .drop(item_code)
                .sort_values(ascending=False)
                .head(top_k)
                .reset_index()
                .rename(columns={"index":"Kode_Item", item_code:"Score"})
                .assign(
                    Nama_Item=lambda x: x["Kode_Item"].map(self.item_lookup),
                    Kandungan=lambda x: x["Kode_Item"].map(self.kandungan_lookup),
                    Method="Content-Based"
                )
            )
        else:
            cb = pd.DataFrame()

        return {
            "co_occurrence": co,
            "content_based": cb
        }

    # def recommend(self, item_code, top_k=5, 
    #             location=None, 
    #             days_back=None):

        # # ---------- Build Context Filter ----------
        # df_ctx = self.df

        # if location is not None:
        #     df_ctx = df_ctx[df_ctx["lokasi_apotek"] == location]

        # if days_back is not None:
        #     cutoff = df_ctx["Tanggal"].max() - pd.Timedelta(days=days_back)
        #     df_ctx = df_ctx[df_ctx["Tanggal"] >= cutoff]

        # valid_items = set(df_ctx["Kode_Item"].unique())

        # # ---------- CO-OCCURRENCE ----------
        # if item_code in self.ibcf_sim:
        #     co = (
        #         self._normalize(self.ibcf_sim[item_code])
        #         .drop(item_code, errors="ignore")
        #         .loc[lambda s: s.index.isin(valid_items)]
        #         .sort_values(ascending=False)
        #         .head(top_k)
        #         .reset_index()
        #         .rename(columns={"index":"Kode_Item", item_code:"Score"})
        #         .assign(
        #             Nama_Item=lambda x: x["Kode_Item"].map(self.item_lookup),
        #             Kandungan=lambda x: x["Kode_Item"].map(self.kandungan_lookup),
        #             Method="Co-Occurrence"
        #         )
        #     )
        # else:
        #     co = pd.DataFrame()

        # # ---------- CONTENT-BASED ----------
        # if item_code in self.content_sim:
        #     cb = (
        #         self._normalize(self.content_sim[item_code])
        #         .drop(item_code)
        #         .loc[lambda s: s.index.isin(valid_items)]
        #         .sort_values(ascending=False)
        #         .head(top_k)
        #         .reset_index()
        #         .rename(columns={"index":"Kode_Item", item_code:"Score"})
        #         .assign(
        #             Nama_Item=lambda x: x["Kode_Item"].map(self.item_lookup),
        #             Kandungan=lambda x: x["Kode_Item"].map(self.kandungan_lookup),
        #             Method="Content-Based"
        #         )
        #     )
        # else:
        #     cb = pd.DataFrame()

        # return {
        #     "co_occurrence": co,
        #     "content_based": cb
        # }
