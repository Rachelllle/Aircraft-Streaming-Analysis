package com.aircraft.analysis

import com.aircraft.analysis.Parsing.{Embeddings, Parsing => ParsingPipeline}

object Main {
  private val logger = AircraftLogger.logger

  def main(args: Array[String]): Unit = {
    val spark = SparkSessionFactory.getSpark()
    try {
      val stage = args.headOption.getOrElse("all")
      logger.info(s"Running stage: $stage")
      stage match {
        case "parse" => ParsingPipeline.buildFullDataset(spark)
        case "embed" => Embeddings.extractEmbeddings(spark)
        case _ =>
          ParsingPipeline.buildFullDataset(spark)
          Embeddings.extractEmbeddings(spark)
      }
    } finally {
      spark.stop()
    }
  }
}
