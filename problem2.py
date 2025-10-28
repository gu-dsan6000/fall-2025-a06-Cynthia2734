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
    min as smin, max as smax, count as s_count, unix_timestamp, desc, lit
)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import gaussian_kde

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
    print("[1/5] Time-series data for each application")
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
    print("[2/5] Aggregated cluster statistics")
    
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
    print("[3/5] Overall summary statistics")
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
    print("[4/5] Bar chart visualization")
    bar_chart_path = os.path.join(output_dir, "problem2_bar_chart.png")
    plt.figure(figsize=(16, 10))
    ax = sns.barplot(
        data=cs_pdf,
        x="cluster_id",
        y="num_applications",
        hue="cluster_id",
        palette="tab20"
    )
    ax.set_xlabel("Cluster ID")
    ax.set_ylabel("Number of Applications")
    ax.set_title("Number of Applications per Cluster")
    ax.set_xticklabels(cs_pdf["cluster_id"].astype(str), rotation=45, ha="right")
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(
            f"{int(height)}",
            (p.get_x() + p.get_width()/2, height),
            ha="center", va="bottom",
            xytext=(0, 3), textcoords="offset points", fontsize=9
        )
    plt.tight_layout()
    plt.savefig(bar_chart_path, dpi=160)
    plt.close()
    print(f"✅ Saved: {bar_chart_path}")

    # 5. Faceted density plot visualization
    logger.info("Faceted density plot visualization")
    print("[5/5] Faceted density plot visualization")
    sns.set_theme(style="whitegrid")
    timeseries = timeseries.withColumn(
        "duration_sec",
        (unix_timestamp(col("end_time")) - unix_timestamp(col("start_time"))).cast("long")
    )
    largest_cluster_id = str(cs_pdf.iloc[0]["cluster_id"])
    dur_pdf = (
        timeseries
        .filter((col("cluster_id") == lit(largest_cluster_id)) &
                col("duration_sec").isNotNull() &
                (col("duration_sec") > 0))
        .select("duration_sec")
        .toPandas()
    )
    dur = dur_pdf["duration_sec"].astype(float).to_numpy()
    dur = dur[np.isfinite(dur)]
    dur = dur[(dur > 0)]
    bins = np.logspace(np.log10(dur.min()), np.log10(dur.max()), 40)
    kde_log = gaussian_kde(np.log10(dur), bw_method="scott")
    log_edges = np.log10(bins)
    logx_grid = np.linspace(log_edges.min(), log_edges.max(), 500)
    delta_log10 = np.mean(np.diff(log_edges))
    counts_curve = kde_log(logx_grid) * len(dur) * delta_log10
    x_grid = 10 ** logx_grid
    density_plot_path = os.path.join(output_dir, "problem2_density_plot.png")

    plt.figure(figsize=(16, 10))
    sns.histplot(dur, bins=bins, stat="count", color="skyblue", edgecolor="black", alpha=0.6)
    plt.plot(x_grid, counts_curve, color="red", linewidth=2)
    plt.xscale("log")
    plt.xlabel("Job Duration (seconds, log scale)")
    plt.ylabel("Count")
    plt.title(f"Duration distribution for Cluster {largest_cluster_id} (n={len(dur)})")
    plt.tight_layout()
    plt.savefig(density_plot_path, dpi=160)
    plt.close()
    print(f"✅ Saved: {density_plot_path}")

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