package com.aircraft.analysis

import org.apache.spark.sql.SparkSession

import java.io.File
import java.nio.file.Files

object SparkSessionFactory {

  private def setupWinutils(): Unit = {
    if (!System.getProperty("os.name", "").toLowerCase.contains("windows")) return
    val winutilsDir = Config.projectRoot.resolve("winutils").toFile
    val binDir      = new File(winutilsDir, "bin")
    binDir.mkdirs()
    System.setProperty("hadoop.home.dir", winutilsDir.getAbsolutePath)
    val winutilsExe = new File(binDir, "winutils.exe")
    if (!winutilsExe.exists()) {
      AircraftLogger.logger.warning(
        s"winutils.exe not found at ${winutilsExe.getAbsolutePath}. " +
        "Download it from https://github.com/cdarlint/winutils (hadoop-3.3.5/bin) " +
        "and place it in the winutils/bin/ folder of this project."
      )
    }
  }

  def getSpark(appName: String = "aircraft_classification_scala"): SparkSession = {
    setupWinutils()
    Config.ensureDirectories()
    AircraftLogger.logger.info("Creating Spark session")

    SparkSession.builder()
      .appName(appName)
      .master(sys.env.getOrElse("SPARK_MASTER", "local[*]"))
      .config("spark.driver.memory", "8g")
      .config("spark.executor.memory", "8g")
      .config("spark.sql.shuffle.partitions", "4")
      .getOrCreate()
  }
}
