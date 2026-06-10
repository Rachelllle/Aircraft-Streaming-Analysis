package com.aircraft.analysis.Predict

import com.aircraft.analysis.{AircraftLogger, Config}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.StringIndexerModel
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.col

import java.nio.charset.StandardCharsets
import java.nio.file.Files

object Scoring {
  private val logger = AircraftLogger.logger

  private def evaluateModel(spark: SparkSession, modelName: String): ModelScore = {
    val model = PipelineModel.load(Config.modelPath.resolve(modelName).toString)
    val testData = spark.read.parquet(Config.rawEmbeddingsPath.toString).filter(col("split") === "test")
    val predictions = model.transform(testData)

    val accuracy = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
      .evaluate(predictions)

    val f1 = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("f1")
      .evaluate(predictions)

    val classes = model.stages.collectFirst {
      case indexer: StringIndexerModel => indexer.labels.length
    }.getOrElse(0)

    logger.info(f"$modelName accuracy: ${accuracy * 100.0}%.2f%%")
    logger.info(f"$modelName f1-score: ${f1 * 100.0}%.2f%%")
    ModelScore(accuracy * 100.0, f1 * 100.0, classes)
  }

  def scoreAllModels(spark: SparkSession): Map[String, ModelScore] = {
    val results = Map(
      "manufacturer" -> evaluateModel(spark, "manufacturer"),
      "family" -> evaluateModel(spark, "family"),
      "variant" -> evaluateModel(spark, "variant")
    )

    val json = ujson.Obj.from(
      results.map { case (name, score) =>
        name -> ujson.Obj(
          "accuracy" -> round(score.accuracy),
          "f1" -> round(score.f1),
          "classes" -> score.classes
        )
      }
    )

    Files.writeString(Config.scoresPath, ujson.write(json, indent = 2), StandardCharsets.UTF_8)
    logger.info(s"Saved scores to ${Config.scoresPath}")
    results
  }

  private def round(value: Double): Double =
    BigDecimal(value).setScale(2, BigDecimal.RoundingMode.HALF_UP).toDouble
}
