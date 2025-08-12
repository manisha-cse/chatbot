
import argparse
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_timestamp, hour, date_format, avg, desc, broadcast

def main(args):
    spark = SparkSession.builder \
        .appName("InternshipTask1_BigDataAnalysis") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .config("spark.sql.shuffle.partitions", "8") \
        .getOrCreate()

    # 1) Read input (CSV or Parquet)
    df = spark.read.option("header", True).option("inferSchema", True).csv(args.input)
    print("Schema:")
    df.printSchema()
    print("Total rows (may be slow):", df.count())

    # 2) Basic cleaning
    df = df.dropDuplicates()
    # change these names to match your dataset
    required_cols = ["pickup_datetime", "dropoff_datetime"]
    df = df.dropna(subset=[c for c in required_cols if c in df.columns])

    if "pickup_datetime" in df.columns:
        df = df.withColumn("pickup_datetime", to_timestamp(col("pickup_datetime")))

    if "dropoff_datetime" in df.columns:
        df = df.withColumn("dropoff_datetime", to_timestamp(col("dropoff_datetime")))

    # 3) Derived columns for analyses
    if "pickup_datetime" in df.columns:
        df = df.withColumn("pickup_hour", hour(col("pickup_datetime"))) \
               .withColumn("pickup_date", date_format(col("pickup_datetime"), "yyyy-MM-dd"))

    # 4) Repartition & cache for repeated queries
    if "pickup_date" in df.columns:
        df = df.repartition(8, "pickup_date")
    df = df.cache()
    df.count()  # materialize cache

    # 5) Example analysis: busiest pickup locations (change column name as needed)
    if "pickup_location_id" in df.columns:
        top_pickups = df.groupBy("pickup_location_id").count().orderBy(desc("count")).limit(20)
        top_pickups.show(20, truncate=False)
        top_pickups.coalesce(1).write.mode("overwrite").csv(f"{args.output}/top_pickups", header=True)
    else:
        print("Column pickup_location_id not found; skipping pickup location analysis.")

    # 6) Example analysis: avg trip distance by hour
    if "trip_distance" in df.columns and "pickup_hour" in df.columns:
        avg_by_hour = df.groupBy("pickup_hour").agg(avg("trip_distance").alias("avg_trip_distance")).orderBy("pickup_hour")
        avg_by_hour.show(24, truncate=False)
        avg_by_hour.coalesce(1).write.mode("overwrite").csv(f"{args.output}/avg_by_hour", header=True)

    # 7) Example broadcast join with small lookup
    if args.zones and "pickup_location_id" in df.columns:
        try:
            zones = spark.read.option("header", True).csv(args.zones)
            df = df.join(broadcast(zones), df.pickup_location_id == zones.location_id, "left")
            df.select("pickup_location_id", "zone").show(5)
        except Exception as e:
            print("Zones join failed:", e)

    # 8) Save cleaned dataset as parquet partitioned by date (fast for later runs)
    if "pickup_date" in df.columns:
        df.write.mode("overwrite").partitionBy("pickup_date").parquet(f"{args.output}/cleaned_parquet")

    spark.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="glob path to input CSV(s) or Parquet(s)")
    parser.add_argument("--zones", required=False, help="optional small lookup CSV for zones")
    parser.add_argument("--output", required=True, help="output folder")
    args = parser.parse_args()
    main(args)
