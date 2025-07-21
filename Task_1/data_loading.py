from pyspark.sql.functions import col

def load_parquet_data(spark, parquet_path: str):
    try:
        df = spark.read.parquet(parquet_path)
        print(f"Loaded Parquet file: {parquet_path}")
        df.printSchema()
        # Show 5 rows with 5 columns only
        df.select(df.columns[:5]).limit(5).show(truncate=100)
        approx_count = df.rdd.map(lambda x: 1).reduce(lambda x, y: x + y)
        print(f"Approximate row count: {approx_count}")
        return df
    except Exception as e:
        print(f"Error loading parquet: {e}")
        return None