import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

DETAILS_PATH = "data/transaction_details_dummy.csv"
ITEMS_PATH = "data/items_dummy.csv"
TRANSACTIONS_PATH = "data/transactions_dummy.csv"

OUTPUT_PATH = "artifacts/ibcf_sim.parquet"

MIN_FREQ = 30     
MAX_QTY_CLIP = 5   

def main():
    print("Loading data...")
    details = pd.read_csv(DETAILS_PATH)
    items = pd.read_csv(ITEMS_PATH)
    transactions = pd.read_csv(TRANSACTIONS_PATH)

    print("Merging data...")
    df = (
        details
        .merge(items, on="Kode_Item", how="left")
        .merge(
            transactions[["Kode_Transaksi", "Tanggal"]],
            on="Kode_Transaksi",
            how="left"
        )
    )

    print("Building basket...")
    basket = (
        df.groupby(["Kode_Transaksi", "Kode_Item"])["qty"]
          .sum()
          .unstack(fill_value=0)
    ).clip(upper=MAX_QTY_CLIP)

    print("Filtering frequent items...")
    item_freq = (basket > 0).sum(axis=0)
    frequent_items = item_freq[item_freq >= MIN_FREQ].index
    basket = basket[frequent_items]

    print(f"Remaining items after filter: {basket.shape[1]}")

    print("Computing cosine similarity (IBCF)...")
    sim_matrix = cosine_similarity(basket.T)

    ibcf_sim = pd.DataFrame(
        sim_matrix,
        index=basket.columns,
        columns=basket.columns
    )

    print("Saving IBCF similarity matrix...")
    ibcf_sim.to_parquet(OUTPUT_PATH)

    print("Precompute finished successfully")
    print(f"File saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
