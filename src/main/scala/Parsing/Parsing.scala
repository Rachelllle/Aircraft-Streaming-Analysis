package com.aircraft.analysis.Parsing

import com.aircraft.analysis.{AircraftLogger, Config}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.IntegerType
import org.apache.spark.sql.{DataFrame, SparkSession}


object Parsing {
  private val logger = AircraftLogger.logger

  def readAnnotationFile(spark: SparkSession, filePath: String, labelName: String): DataFrame =
    spark.read.text(filePath).select(
      split(col("value"), " ", 2).getItem(0).as("image_id"),
      split(col("value"), " ", 2).getItem(1).as(labelName)
    )

  def buildAnnotationDf(spark: SparkSession, splitName: String): DataFrame = {
    val manufacturer = readAnnotationFile(
      spark,
      Config.dataPath.resolve(s"images_manufacturer_$splitName.txt").toString,
      "manufacturer"
    )
    val family = readAnnotationFile(
      spark,
      Config.dataPath.resolve(s"images_family_$splitName.txt").toString,
      "family"
    )
    val variant = readAnnotationFile(
      spark,
      Config.dataPath.resolve(s"images_variant_$splitName.txt").toString,
      "variant"
    )

    manufacturer
      .join(family, Seq("image_id"))
      .join(variant, Seq("image_id"))
      .withColumn("split", lit(splitName))
  }

  def parseAnnotations(spark: SparkSession): DataFrame = {
    logger.info("Combining train, val and test annotations")
    val annotations = buildAnnotationDf(spark, "train")
      .unionByName(buildAnnotationDf(spark, "val"))
      .unionByName(buildAnnotationDf(spark, "test"))

    logger.info(s"Annotations loaded: ${annotations.count()} rows")
    annotations
  }

  def parseBoundingBoxes(spark: SparkSession): DataFrame =
    spark.read.text(Config.dataPath.resolve("images_box.txt").toString)
      .select(
        split(col("value"), " ").getItem(0).as("image_id"),
        split(col("value"), " ").getItem(1).cast(IntegerType).as("bbox_x1"),
        split(col("value"), " ").getItem(2).cast(IntegerType).as("bbox_y1"),
        split(col("value"), " ").getItem(3).cast(IntegerType).as("bbox_x2"),
        split(col("value"), " ").getItem(4).cast(IntegerType).as("bbox_y2")
      )

  def buildFullDataset(spark: SparkSession): DataFrame = {
    logger.info("Loading image files")
    val images = spark.read.format("binaryFile")
      .option("pathGlobFilter", "*.jpg")
      .load(Config.dataPath.resolve("images").toString)
      .withColumn("image_id", regexp_extract(
        element_at(split(col("path"), "[/\\\\]"), -1), "(.+)\\.jpg", 1
      ))
      .select("image_id", "path")

    val annotations = parseAnnotations(spark)
    val bboxes      = parseBoundingBoxes(spark)

    val parsed = images
      .join(annotations, Seq("image_id"))
      .join(bboxes, Seq("image_id"), "left")

    parsed.write.mode("overwrite").parquet(Config.parsedDatasetPath.toString)
    logger.info(s"Parsing complete: ${parsed.count()} rows saved to ${Config.parsedDatasetPath}")
    parsed.show(5, truncate = false)
    parsed.groupBy("split").count().show()
    parsed.select("manufacturer").distinct().show(10, truncate = false)
    parsed
  }
}
