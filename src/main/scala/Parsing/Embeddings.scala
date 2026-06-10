package com.aircraft.analysis.Parsing

import com.aircraft.analysis.{AircraftLogger, Config}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.{DataFrame, SparkSession}

import java.awt.image.BufferedImage
import java.io.File
import java.net.URI
import java.nio.file.Paths
import javax.imageio.ImageIO

object Embeddings {
  private val logger = AircraftLogger.logger

  private val vectorLength = Config.imgSize * Config.imgSize * 3

  def extractPixelFeatures(imagePath: String): Array[Double] = {
    try {
      val file =
        if (imagePath.startsWith("file:")) Paths.get(URI.create(imagePath)).toFile
        else new File(imagePath)

      val source = ImageIO.read(file)
      if (source == null) {
        Array.fill(vectorLength)(0.0)
      } else {
        val resized = new BufferedImage(Config.imgSize, Config.imgSize, BufferedImage.TYPE_INT_RGB)
        val graphics = resized.createGraphics()
        graphics.setRenderingHint(
          java.awt.RenderingHints.KEY_INTERPOLATION,
          java.awt.RenderingHints.VALUE_INTERPOLATION_BILINEAR
        )
        graphics.drawImage(source, 0, 0, Config.imgSize, Config.imgSize, null)
        graphics.dispose()

        val pixels = new Array[Double](vectorLength)
        var index = 0
        var y = 0
        while (y < Config.imgSize) {
          var x = 0
          while (x < Config.imgSize) {
            val rgb = resized.getRGB(x, y)
            pixels(index) = ((rgb >> 16) & 0xff) / 255.0
            pixels(index + 1) = ((rgb >> 8) & 0xff) / 255.0
            pixels(index + 2) = (rgb & 0xff) / 255.0
            index += 3
            x += 1
          }
          y += 1
        }
        pixels
      }
    } catch {
      case _: Throwable => Array.fill(vectorLength)(0.0)
    }
  }

  val imagePathToVector: UserDefinedFunction =
    org.apache.spark.sql.functions.udf((imagePath: String) => Vectors.dense(extractPixelFeatures(imagePath)))

  def addRawFeatures(dataFrame: DataFrame, pathColumn: String = "path"): DataFrame =
    dataFrame.withColumn("features_raw", imagePathToVector(col(pathColumn)))

  def extractEmbeddings(spark: SparkSession): DataFrame = {
    logger.info("Loading parsed metadata")
    val parsed = spark.read.parquet(Config.parsedDatasetPath.toString)
    val rawEmbeddings = addRawFeatures(parsed)
    rawEmbeddings.write.mode("overwrite").parquet(Config.rawEmbeddingsPath.toString)
    logger.info(s"Raw image vectors saved to ${Config.rawEmbeddingsPath}")
    rawEmbeddings
  }

  def loadRawFeatures(spark: SparkSession): DataFrame =
    spark.read.parquet(Config.rawEmbeddingsPath.toString)
}
