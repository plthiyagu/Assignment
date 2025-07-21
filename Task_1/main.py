import os
import pandas as pd

from setup_env import mount_drive, create_spark_session
from data_loading import load_parquet_data
from data_preparation import prepare_edgar_dataset
from text_chunking import create_chunked_dataframe
from embedding_utils import get_sentence_embeddings, standard_scale_embeddings, reduce_dimensions
from clustering import assign_kmeans_clusters, add_outlier_flag
from plotting import plot_and_save_embeddings_no_section

def main():
    # Setup and Spark
    mount_drive()
    spark = create_spark_session()

    # Paths
    year = 2020
    output_dir = "/content/drive/My Drive/edgar_data"
    parquet_path = os.path.join(output_dir, f"edgar_pandas_{year}.parquet")

    # Load data
    df = load_parquet_data(spark, parquet_path)
    if df is None:
        print("Data loading failed. Exiting.")
        return

    # Prepare data
    df_prepared = prepare_edgar_dataset(df, year=year, top_n=10)
    print(f"Filtered and prepared dataset count: {df_prepared.count()}")

    # Convert to pandas
    pdf = df_prepared.select("cik", "filename", "full_text").toPandas()
    print(f"Pandas DataFrame memory usage: {pdf.memory_usage(deep=True).sum() / (1024**2):.2f} MB")

    # Chunk text
    df_chunks = create_chunked_dataframe(pdf, chunk_size=1000, overlap=200)

    # Embeddings
    texts = df_chunks['chunk_text'].tolist()
    embeddings = get_sentence_embeddings(texts, model_name='all-MiniLM-L6-v2')
    df_chunks['embedding'] = list(embeddings)

    # Scale embeddings
    df_chunks = standard_scale_embeddings(df_chunks)

    # Dimensionality reduction (PCA)
    df_chunks = reduce_dimensions(df_chunks, method='pca', col_prefix='pca')

    # Clustering
    df_chunks = assign_kmeans_clusters(df_chunks, n_clusters=5)
    df_chunks = add_outlier_flag(df_chunks, threshold_std=2.0)

    # Plot results
    plot_and_save_embeddings_no_section(df_chunks, x_col='pca_0', y_col='pca_1')

    # Show dataframe columns for reference
    print(df_chunks.columns.tolist())

if __name__ == '__main__':
    main()