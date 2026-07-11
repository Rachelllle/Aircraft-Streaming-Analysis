import java.io.{ByteArrayInputStream, File, PrintWriter}
import javax.imageio.ImageIO

import scala.sys.process._

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.streaming.Trigger
import org.apache.spark.sql.types._

object Consumer {

  case class Pixels(largeur: Int, hauteur: Int, hist: Seq[Double])

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

    val pixelsUdf = udf { (content: Array[Byte]) =>
      try {
        val img = ImageIO.read(new ByteArrayInputStream(content))
        if (img == null) Pixels(0, 0, Seq.fill(16)(0.0))
        else {
          val hist = new Array[Double](16)
          var n = 0
          var y = 0
          while (y < img.getHeight) {
            var x = 0
            while (x < img.getWidth) {
              val rgb = img.getRGB(x, y)
              val gris = (((rgb >> 16) & 0xFF) + ((rgb >> 8) & 0xFF) + (rgb & 0xFF)) / 3
              hist(gris / 16) += 1
              n += 1
              x += 4
            }
            y += 4
          }
          Pixels(img.getWidth, img.getHeight, hist.map(_ / n).toSeq)
        }
      } catch {
        case _: Exception => Pixels(0, 0, Seq.fill(16)(0.0))
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
      .withColumn("px", pixelsUdf(col("content")))
      .select(
        Seq(
          col("image"),
          col("classe"),
          col("taille"),
          col("moyenne_octets"),
          col("px.largeur").as("largeur"),
          col("px.hauteur").as("hauteur")
        ) ++ (0 until 16).map(i => col("px.hist").getItem(i).as(s"hist_$i")): _*
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
      resultat
        .select("image", "classe", "taille", "moyenne_octets", "largeur", "hauteur")
        .writeStream
        .format("console")
        .outputMode("append")
        .option("truncate", false)
        .option("checkpointLocation", s"${config.checkpointPath}/console")
        .trigger(Trigger.ProcessingTime(s"${config.streamInterval} seconds"))
        .start()

      resultat.writeStream
        .format("csv")
        .outputMode("append")
        .option("header", "true")
        .option("path", config.outputPath)
        .option("checkpointLocation", s"${config.checkpointPath}/csv")
        .trigger(Trigger.ProcessingTime(s"${config.streamInterval} seconds"))
        .start()
    }

    spark.streams.awaitAnyTermination()
  }

  def predire(spark: SparkSession, config: AppConfig, batchDF: DataFrame, batchId: Long): Unit = {
    val lignes = batchDF.collect()
    if (lignes.isEmpty) return

    val tmpDir = new File("data/tmp")
    tmpDir.mkdirs()
    val csvEntree = new File(tmpDir, s"batch_$batchId.csv")
    val csvSortie = new File(tmpDir, s"pred_$batchId.csv")

    val pw = new PrintWriter(csvEntree)
    pw.println(batchDF.columns.mkString(","))
    lignes.foreach { l =>
      pw.println(l.toSeq.mkString(","))
    }
    pw.close()

    val code = Seq("python", "ml-service/predict.py",
      config.modelPath, csvEntree.getPath, csvSortie.getPath).!

    if (code == 0 && csvSortie.exists()) {
      val predictions = spark.read.option("header", "true").csv(csvSortie.getPath)
      println(s"===== Batch $batchId : ${lignes.length} predictions =====")
      predictions.show(lignes.length, truncate = false)
      predictions.write.mode("append").option("header", "true").csv(config.outputPath)
    } else {
      println(s"Erreur de prediction pour le batch $batchId (code $code)")
    }
  }
}
