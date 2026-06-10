package com.aircraft.analysis.Predict

import com.aircraft.analysis.Parsing.Embeddings
import com.aircraft.analysis.{AircraftLogger, Config}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.feature.StringIndexerModel
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.lit

import java.nio.file.Path

object Predict {
  private val logger = AircraftLogger.logger

  final case class RuntimeBundle(
      models: Map[String, PipelineModel]
  )

  def loadRuntime(spark: SparkSession): RuntimeBundle = {
    logger.info("Loading RF classifier models")
    RuntimeBundle(
      models = Seq("manufacturer", "family", "variant")
        .map(name => name -> PipelineModel.load(Config.modelPath.resolve(name).toString))
        .toMap
    )
  }

  def predictImage(
      spark: SparkSession,
      imagePath: Path,
      runtimeOverride: Option[RuntimeBundle] = None
  ): Map[String, PredictionSummary] = {
    import spark.implicits._

    val runtime = runtimeOverride.getOrElse(loadRuntime(spark))
    val imageId = imagePath.getFileName.toString.replaceAll("\\.jpg$$", "")

    val singleImage = Seq((imageId, imagePath.toAbsolutePath.toString)).toDF("image_id", "path")
    val rawFeatures = Embeddings.addRawFeatures(singleImage)

    runtime.models.map { case (target, pipelineModel) =>
      val features = rawFeatures.withColumn(target, lit(""))
      val resultRow = pipelineModel.transform(features).select("prediction", "probability").head()

      val predictionIndex = resultRow.getDouble(0).toInt
      val probabilities = resultRow.getAs[Vector]("probability").toArray

      val labels = pipelineModel.stages.collectFirst {
        case indexer: StringIndexerModel => indexer.labels.toSeq
      }.getOrElse(Seq.empty)

      val label = labels.lift(predictionIndex).getOrElse(predictionIndex.toString)
      val ranked = probabilities.zipWithIndex
        .sortBy { case (probability, _) => -probability }
        .take(3)
        .map { case (probability, index) =>
          RankedPrediction(labels.lift(index).getOrElse(index.toString), probability * 100.0)
        }
        .toSeq

      target -> PredictionSummary(label, probabilities(predictionIndex) * 100.0, ranked)
    }
  }
}
