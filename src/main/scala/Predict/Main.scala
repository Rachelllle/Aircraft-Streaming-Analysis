package com.aircraft.analysis.Predict

import com.aircraft.analysis.{AircraftLogger, SparkSessionFactory}

object PredictMain {
  private val logger = AircraftLogger.logger

  def main(args: Array[String]): Unit = {
    val spark = SparkSessionFactory.getSpark()
    try {
      logger.info("Running scoring")
      Scoring.scoreAllModels(spark)
    } finally {
      spark.stop()
    }
  }
}
