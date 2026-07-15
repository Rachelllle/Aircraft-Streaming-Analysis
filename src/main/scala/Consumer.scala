import java.io.{ByteArrayInputStream, File, PrintWriter}
import javax.imageio.ImageIO

import scala.sys.process._

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.streaming.Trigger
import org.apache.spark.sql.types._

object Consumer {

  case class Pixels(largeur: Int, hauteur: Int, grille: Seq[Double], contours: Double, moyennePixels: Double, ratio: Double)

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

    val gridSize = 8
    val featuresParCellule = 6 // moyenne gris, ecart-type, R, G, B, contours
    val nbFeatures = gridSize * gridSize * featuresParCellule // 384

    val pixelsUdf = udf { (content: Array[Byte]) =>
      try {
        val img = ImageIO.read(new ByteArrayInputStream(content))
        if (img == null) Pixels(0, 0, Seq.fill(nbFeatures)(0.0), 0.0, 0.0, 0.0)
        else {
          val w = img.getWidth; val h = img.getHeight
          val cellW = math.max(1, w / gridSize)
          val cellH = math.max(1, h / gridSize)
          val feats = new Array[Double](nbFeatures)

          var sommeGlobale = 0.0
          var nGlobal = 0
          var contoursGlobal = 0

          for (cy <- 0 until gridSize; cx <- 0 until gridSize) {
            val xs = cx * cellW; val ys = cy * cellH
            val xe = if (cx == gridSize - 1) w else xs + cellW
            val ye = if (cy == gridSize - 1) h else ys + cellH

            var sum = 0.0; var sumSq = 0.0
            var sumR = 0.0; var sumG = 0.0; var sumB = 0.0
            var contours = 0; var n = 0

            var y = ys
            while (y < ye) {
              var precedent = -1.0 // reinitialise a chaque ligne
              var x = xs
              while (x < xe) {
                val rgb = img.getRGB(x, y)
                val r = (rgb >> 16) & 0xFF
                val g = (rgb >> 8) & 0xFF
                val b = rgb & 0xFF
                val gris = (r + g + b) / 3.0
                sum += gris; sumSq += gris * gris
                sumR += r; sumG += g; sumB += b
                n += 1
                sommeGlobale += gris; nGlobal += 1
                if (precedent >= 0 && math.abs(gris - precedent) > 25) {
                  contours += 1; contoursGlobal += 1
                }
                precedent = gris
                x += 2
              }
              y += 2
            }

            val idx = (cy * gridSize + cx) * featuresParCellule
            val moy = if (n > 0) sum / n else 0.0
            feats(idx)     = moy
            feats(idx + 1) = if (n > 0) math.sqrt(math.max(0, sumSq / n - moy * moy)) else 0.0
            feats(idx + 2) = if (n > 0) sumR / n else 0.0
            feats(idx + 3) = if (n > 0) sumG / n else 0.0
            feats(idx + 4) = if (n > 0) sumB / n else 0.0
            feats(idx + 5) = if (n > 0) contours.toDouble / n else 0.0
          }

          val moyennePixels = if (nGlobal > 0) sommeGlobale / nGlobal else 0.0
          val densiteContours = if (nGlobal > 0) contoursGlobal.toDouble / nGlobal else 0.0
          val ratio = if (h > 0) w.toDouble / h else 0.0

          Pixels(w, h, feats.toSeq, densiteContours, moyennePixels, ratio)
        }
      } catch {
        case _: Exception => Pixels(0, 0, Seq.fill(nbFeatures)(0.0), 0.0, 0.0, 0.0)
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
      .withColumn("px", pixelsUdf(col("content")))
      .select(
        Seq(
          col("image"),
          col("classe"),
          col("taille"),
          col("px.moyennePixels").as("moyenne_pixels"),
          col("px.contours").as("densite_contours"),
          col("px.largeur").as("largeur"),
          col("px.hauteur").as("hauteur"),
          col("px.ratio").as("ratio")
        ) ++ (0 until nbFeatures).map(i => col("px.grille").getItem(i).as(s"grille_$i")): _*
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
        .select("image", "classe", "taille", "moyenne_pixels", "densite_contours", "largeur", "hauteur")
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
