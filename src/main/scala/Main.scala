package com.aircraft.analysis

import com.aircraft.analysis.Parsing.{Embeddings, Parsing => ParsingPipeline}
import com.aircraft.analysis.Predict.Scoring

object Main {
  private val logger = AircraftLogger.logger

  def main(args: Array[String]): Unit = {
    val spark = SparkSessionFactory.getSpark()
    try {
      val stage = args.headOption.getOrElse("all")
      logger.info(s"Running stage: $stage")
      stage match {
        case "parse" =>
          ParsingPipeline.buildFullDataset(spark)
        case "embed" =>
          Embeddings.extractEmbeddings(spark)
        case "parsing" | "all" =>
          ParsingPipeline.buildFullDataset(spark)
          Embeddings.extractEmbeddings(spark)
        case "score" =>
          Scoring.scoreAllModels(spark)
        case other =>
          throw new IllegalArgumentException(
            s"Unknown stage: $other. Supported stages: parse, embed, all, score."
          )
      }
    } finally {
      spark.stop()
    }
  }
}
