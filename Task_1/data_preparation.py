from pyspark.sql.functions import concat_ws, col

def prepare_edgar_dataset(df, year=2020, top_n=10):
    # Filter by year
    df_year = df.filter(col("year") == year)

    # Get top N companies by distinct CIKs
    top_companies = df_year.select("cik").distinct().limit(top_n)
    top_cik_list = [row["cik"] for row in top_companies.collect()]

    # Filter to top companies
    df_top = df_year.filter(col("cik").isin(top_cik_list))

    # Define section columns to combine
    section_cols = [f"section_{i}" for i in range(1, 16)] + ["section_1A", "section_1B", "section_7A", "section_9A", "section_9B"]
    available_sections = [c for c in section_cols if c in df.columns]

    # Combine all sections into a single text column
    df_combined = df_top.withColumn(
        "full_text",
        concat_ws(" ", *[col(c) for c in available_sections])
    )
    return df_combined