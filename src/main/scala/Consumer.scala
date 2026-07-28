import java.io.{ByteArrayInputStream, File}
import javax.imageio.ImageIO

import scala.sys.process._

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.streaming.Trigger
import org.apache.spark.sql.types._

object Consumer {

  case class Dimensions(largeur: Int, hauteur: Int, ratio: Double)

  def main(args: Array[String]): Unit = {
    val config = AppConfig.load()

    val spark = SparkSession.builder()
      .appName("ImageConsumer")
      .master("local[*]")
      .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    val labelMap: Map[String, String] = {
      val root = new File(config.inputPath)
      if (root.exists()) {
        root.listFiles().filter(_.isDirectory).flatMap { dossier =>
          dossier.listFiles().filter(_.isFile).map(f => f.getName -> dossier.getName)
        }.toMap
      } else Map.empty
    }
    val labelUdf = udf((nom: String) => labelMap.getOrElse(nom, "inconnu"))

    val moyenneUdf = udf { (content: Array[Byte]) =>
      if (content == null || content.isEmpty) 0.0
      else content.map(b => (b & 0xFF).toDouble).sum / content.length
    }

    val dimensionsUdf = udf { (content: Array[Byte]) =>
      try {
        val img = ImageIO.read(new ByteArrayInputStream(content))
        if (img == null) Dimensions(0, 0, 0.0)
        else {
          val w = img.getWidth
          val h = img.getHeight
          Dimensions(w, h, if (h > 0) w.toDouble / h else 0.0)
        }
      } catch {
        case _: Exception => Dimensions(0, 0, 0.0)
      }
    }
    val schema = StructType(Seq(
      StructField("path", StringType, nullable = false),
      StructField("modificationTime", TimestampType, nullable = false),
      StructField("length", LongType, nullable = false),
      StructField("content", BinaryType, nullable = true)
    ))

    val flux = spark.readStream
      .format("binaryFile")
      .schema(schema)
      .option("pathGlobFilter", "*.{jpg,JPG,png,PNG}")
      .option("maxFilesPerTrigger", config.batchSize)
      .load(config.intermediatePath)

    val resultat = flux
      .withColumn("image", regexp_extract(col("path"), "([^/\\\\]+)$", 1))
      .withColumn("classe", labelUdf(col("image")))
      .withColumn("taille", col("length"))
      .withColumn("moyenne_octets", moyenneUdf(col("content")))
      .withColumn("dims", dimensionsUdf(col("content")))
      .select(
        col("image"),
        col("classe"),
        col("taille"),
        col("moyenne_octets"),
        col("dims.largeur").as("largeur"),
        col("dims.hauteur").as("hauteur"),
        col("dims.ratio").as("ratio")
      )
    if (config.mode == "predict") {
      resultat.writeStream
        .foreachBatch { (batchDF: DataFrame, batchId: Long) =>
          predire(spark, config, batchDF, batchId)
        }
        .option("checkpointLocation", s"${config.checkpointPath}/predict")
        .trigger(Trigger.ProcessingTime(s"${config.streamInterval} seconds"))
        .start()
    } else {
      resultat.writeStream
        .format("console")
        .outputMode("append")
        .option("truncate", false)
        .option("checkpointLocation", s"${config.checkpointPath}/console")
        .trigger(Trigger.ProcessingTime(s"${config.streamInterval} seconds"))
        .start()

      resultat.writeStream
        .format("parquet")
        .outputMode("append")
        .option("path", config.outputPath)
        .option("checkpointLocation", s"${config.checkpointPath}/parquet")
        .trigger(Trigger.ProcessingTime(s"${config.streamInterval} seconds"))
        .start()
    }

    spark.streams.awaitAnyTermination()
  }

  def predire(spark: SparkSession, config: AppConfig, batchDF: DataFrame, batchId: Long): Unit = {
    if (batchDF.isEmpty) return

    val tmpDir = new File("data/tmp")
    tmpDir.mkdirs()
    val entree = new File(tmpDir, s"batch_$batchId.parquet").getPath
    val sortie = new File(tmpDir, s"pred_$batchId.parquet").getPath

    batchDF.write.mode("overwrite").parquet(entree)

    val code = Seq("python", "ml-service/predict.py",
      config.modelPath, entree, sortie).!

    if (code == 0) {
      val predictions = spark.read.parquet(sortie)
      println(s"===== Batch $batchId : ${predictions.count()} predictions =====")
      predictions.show(truncate = false)
      predictions.write.mode("append").parquet(config.outputPath)
    } else {
      println(s"Erreur de prediction pour le batch $batchId (code $code)")
    }
  }
}
