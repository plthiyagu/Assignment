import os
import warnings
import psutil
import findspark

# Suppress warnings globally
warnings.filterwarnings('ignore')

# Initialize findspark
findspark.init()

from pyspark.sql import SparkSession
from google.colab import drive

def mount_drive():
    drive.mount('/content/drive')

def create_spark_session(app_name="EDGARAnalysis", driver_memory="6g", max_result_size="2g", shuffle_partitions=100):
    spark = SparkSession.builder \
        .appName(app_name) \
        .config("spark.driver.memory", driver_memory) \
        .config("spark.driver.maxResultSize", max_result_size) \
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions)) \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .getOrCreate()

    print(f"PySpark Session created: driver.memory={driver_memory}, shuffle.partitions={shuffle_partitions}")
    print(f"Available RAM: {round(psutil.virtual_memory().available / 1e9, 2)} GB")
    return spark