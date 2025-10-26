#!/usr/bin/env python3
"""
Problem 1: Log Level Distribution
"""

import os
import sys
import time
import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, regexp_extract, rand, when, lit, desc, count as s_count, format_number
)
from pyspark.sql.types import LongType
import pandas as pd

# Configure logging with basicConfig
logging.basicConfig(
    level=logging.INFO,  # Set the log level to INFO
    # Define log message format
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

logger = logging.getLogger(__name__)


def create_spark_session(master_url):
    """Create a Spark session optimized for cluster execution."""

    spark = (
        SparkSession.builder
        .appName("Problem1_LogLevelDist")

        # Cluster Configuration
        .master(master_url)  # Connect to Spark cluster

        # Memory Configuration
        .config("spark.executor.memory", "4g")
        .config("spark.driver.memory", "4g")
        .config("spark.driver.maxResultSize", "2g")

        # Executor Configuration
        .config("spark.executor.cores", "2")
        .config("spark.cores.max", "6")  # Use all available cores across cluster

        # S3 Configuration - Use S3A for AWS S3 access
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.aws.credentials.provider", "com.amazonaws.auth.InstanceProfileCredentialsProvider")

        # Performance settings for cluster execution
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")

        # Serialization
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")

        # Arrow optimization for Pandas conversion
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")

        .getOrCreate()
    )

    logger.info("Spark session created successfully for cluster execution")
    return spark


def run_problem1(spark: SparkSession) -> None:
    """
    Reads all parquet files under input_path and writes three CSV outputs to output_dir.
    """
    start_time = time.time()
    input_path = "s3a://qw172-assignment-spark-cluster-logs/data/"
    output_dir = os.path.expanduser("~/spark-cluster")
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Reading raw log data from: {input_path}")

    print("\nReading raw log data...")
    print("=" * 60)
    print(f"Input path: {input_path}")
    df = spark.read.text(input_path + "/**/*")
    df = df.withColumnRenamed("value", "log_entry")
    df.show(5, truncate=False)

    # 1. Count log levels
    logger.info("Count log levels")
    print("[1/3] Count log levels")
    LEVELS = ["INFO", "WARN", "ERROR", "DEBUG"]
    LEVEL_RE = r"\b(INFO|WARN|ERROR|DEBUG)\b"
    df = df.withColumn("log_level", regexp_extract(col("log_entry"), LEVEL_RE, 1))
    df_lv = df.where(col("log_level").isin(LEVELS))
    levels_df = spark.createDataFrame([(lv,) for lv in LEVELS], ["log_level"])
    counts_base = df_lv.groupBy("log_level").agg(s_count("*").alias("count"))
    counts_df = levels_df.join(counts_base, on="log_level", how="left").na.fill({"count": 0}).withColumn("count", col("count").cast(LongType()))
    order_col = when(col("log_level")=="INFO", lit(0)) \
              .when(col("log_level")=="WARN", lit(1)) \
              .when(col("log_level")=="ERROR", lit(2)) \
              .when(col("log_level")=="DEBUG", lit(3)) \
              .otherwise(lit(9))
    counts_df = counts_df.orderBy(order_col)
    counts_path = os.path.join(output_dir, "problem1_counts.csv")
    counts_df.toPandas().to_csv(counts_path, index=False)
    counts_df.show()

    # 2. Save 10 random sample log entries with their levels
    logger.info("Save 10 random sample log entries with their levels")
    print("[2/3] Save 10 random sample log entries with their levels")
    sample_df = df_lv.select("log_entry", "log_level").orderBy(rand()).limit(10)
    sample_path = os.path.join(output_dir, "problem1_sample.csv")
    sample_df.toPandas().to_csv(sample_path, index=False)
    sample_df.show()

    # 3. Summary statistics
    logger.info("Summary statistics")
    print("[3/3] Summary statistics")
    total_rows = df.count()
    total_with_level = df_lv.count()
    counts = {r["log_level"]: int(r["count"]) for r in counts_df.collect()}
    total_log_lv = sum(1 for lv in LEVELS if counts.get(lv, 0) > 0)
    summary = [f"Total log lines processed: {total_rows:,}", 
               f"Total lines with log levels: {total_with_level:,}", 
               f"Unique log levels found: {total_log_lv:,}",
               f"\nLog level distribution:"]

    def pct(n, d): return (n / d * 100.0) if d else 0.0
    
    for lv in LEVELS:
        n = counts.get(lv, 0)
        line = pct(n, total_with_level)
        summary.append(f"  {lv:<5}: {n:>10,} ({line:5.2f}%)")
    summary_text = "\n".join(summary)
    summary_path = os.path.join(output_dir, "problem1_summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary_text)
    print(summary_text)


    # Calculate execution time
    end_time = time.time()
    execution_time = end_time - start_time
    logger.info(f"Problem 3 execution completed in {execution_time:.2f} seconds")
    

def main():
    if len(sys.argv) < 2:
        print("❌ Usage: python problem1.py spark://MASTER_PRIVATE_IP:7077")
        return 1
    master_url = sys.argv[1]
    spark = create_spark_session(master_url)
    try:
        run_problem1(spark)
    except Exception as e:
        logger.exception(f"Error during Log Level Distribution analysis: {e}")
        print(f"❌ Error: {e}")
        return 1
    finally:
        spark.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())