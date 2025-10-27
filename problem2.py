#!/usr/bin/env python3
"""
Problem 2: Cluster Usage Analysis
"""

import os
import sys
import time
import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, input_file_name, regexp_extract, when, length, to_timestamp, lpad,
    min as smin, max as smax, count as s_count, unix_timestamp, desc
)
import matplotlib.pyplot as plt
import numpy as np

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
        .appName("Problem2_ClusterUsageAnalysis")

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


def run_problem2(spark: SparkSession) -> None:
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

    logs_df = spark.read.text(input_path + "/**/*").withColumnRenamed("value", "line")
    df = logs_df.withColumn('path', input_file_name())
    df = (df
            .withColumn("application_id",
                        regexp_extract(col("path"), r"(application_\d+_\d+)", 1))
            .withColumn("cluster_id",
                        regexp_extract(col("path"), r"application_(\d+)_\d+", 1))
            .withColumn("app_number",
                        regexp_extract(col("path"), r"application_\d+_(\d+)", 1))
            )
    df.show(5, truncate=False)

    # 1. Time-series data for each application
    logger.info("Time-series data for each application")
    print("[1/3] Time-series data for each application")
    cand = regexp_extract(col("line"), r"(\d{2}/\d{2}/\d{2} \d{2}:\d{2}:\d{2})", 1)
    ts = to_timestamp(when(length(cand) > 0, cand), "yy/MM/dd HH:mm:ss")
    logs = df.withColumn("ts", ts)
    logs_ts = logs.where(col("ts").isNotNull())
    timeseries = (logs_ts
            .groupBy("cluster_id", "application_id", "app_number")
            .agg(smin("ts").alias("start_time"),
                 smax("ts").alias("end_time")))
    timeseries_path = os.path.join(output_dir, "problem2_timeline.csv")
    timeseries.toPandas().to_csv(timeseries_path, index=False)
    timeseries.show(5, truncate=False)

    # 2. Aggregated cluster statistics
    logger.info("Aggregated cluster statistics")
    print("[2/3] Aggregated cluster statistics")
    
    cluster_summary = (
        timeseries.groupBy("cluster_id")
                .agg(
                    s_count("*").alias("num_applications"),
                    smin("start_time").alias("cluster_first_app"),
                    smax("end_time").alias("cluster_last_app"),
                )
                .orderBy(desc("num_applications"))
    )
    cluster_summary_path = os.path.join(output_dir, "problem2_cluster_summary.csv")
    cluster_summary.toPandas().to_csv(cluster_summary_path, index=False)
    cluster_summary.show()

    # 3. Overall summary statistics
    logger.info("Overall summary statistics")
    print("[3/3] Overall summary statistics")
    total_clusters = cluster_summary.count()
    total_apps = timeseries.count()
    avg_apps_per_cluster = (total_apps / total_clusters) if total_clusters else 0.0
    summary = [f"Total unique clusters: {total_clusters:,}", 
               f"Total applications: {total_apps:,}", 
               f"Average applications per cluster: {avg_apps_per_cluster:.2f}",
               f"\nMost heavily used clusters:"]
    cs_pdf = (cluster_summary
          .orderBy(col("num_applications").desc())
          .select("cluster_id", "num_applications")
          .toPandas())
    for _, row in cs_pdf.iterrows():
        summary.append(f"  Cluster {row['cluster_id']}: {int(row['num_applications'])} applications")
    summary_text = "\n".join(summary)
    summary_path = os.path.join(output_dir, "problem2_stats.txt")
    with open(summary_path, "w") as f:
        f.write(summary_text)
    print(summary_text)

    # 4. Bar chart visualization
    logger.info("Bar chart visualization")
    print("4/5] Bar chart visualization")
    bar_chart_path = os.path.join(output_dir, "problem2_bar_chart.png")
    plt.figure(figsize=(10, 16))
    x = np.arange(len(cs_pdf))
    y = cs_pdf["num_applications"].to_numpy()
    cmap = plt.get_cmap("tab20")
    colors = cmap(np.linspace(0, 1, len(cs_pdf)))
    plt.bar(x, y, color=colors)
    plt.xlabel("Cluster ID")
    plt.ylabel("Number of Applications")
    plt.title("Number of Applications per Cluster")
    plt.xticks(x, cs_pdf["cluster_id"].astype(str), rotation=45, ha="right")
    ax = plt.gca()
    for xi, yi in zip(x, y):
        ax.annotate(f"{int(yi)}",
                    (xi, yi),
                    ha="center", va="bottom",
                    xytext=(0, 3), textcoords="offset points",
                    fontsize=9)
    plt.tight_layout()
    plt.savefig(bar_chart_path, dpi=160)
    plt.close()

    # 5. Faceted density plot visualization
    logger.info("Faceted density plot visualization")
    print("5/5] Faceted density plot visualization")
    

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
        run_problem2(spark)
    except Exception as e:
        logger.exception(f"Error during Log Level Distribution analysis: {e}")
        print(f"❌ Error: {e}")
        return 1
    finally:
        spark.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())