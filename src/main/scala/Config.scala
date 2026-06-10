package com.aircraft.analysis

import java.nio.file.{Files, Path, Paths}

object Config {
  private val projectRootValue = Paths.get("").toAbsolutePath.normalize()
  private val parentRoot = Option(projectRootValue.getParent).getOrElse(projectRootValue)
  private val siblingPythonProject = parentRoot.resolve("Aircraft-Analysis-Engine").normalize()

  private def resolveExisting(candidates: Seq[Path]): Path =
    candidates.find(Files.exists(_)).getOrElse(candidates.head)

  val projectRoot: Path = projectRootValue
  val basePath: Path = resolveExisting(
    Seq(
      projectRoot.resolve("data").resolve("aircraft_data"),
      siblingPythonProject.resolve("data").resolve("aircraft_data")
    )
  )
  val dataPath: Path = basePath
    .resolve("fgvc-aircraft-2013b")
    .resolve("fgvc-aircraft-2013b")
    .resolve("data")
  val csvPath: Path = basePath
  val parsingOutputPath: Path = projectRoot.resolve("data").resolve("parsing_output")
  val modelPath: Path         = projectRoot.resolve("data").resolve("models")
  val outputPath: Path        = projectRoot.resolve("outputs")
  val logPath: Path           = projectRoot.resolve("logs")

  val parsedDatasetPath: Path = parsingOutputPath.resolve("parsed_dataset.parquet")
  val rawEmbeddingsPath: Path = parsingOutputPath.resolve("raw_embeddings.parquet")
  val scoresPath: Path        = outputPath.resolve("scores.json")
  val predictionsPath: Path   = outputPath.resolve("predictions.json")
  val tempPredictionPath: Path = outputPath.resolve("temp_predict.jpg")

  val imgSize: Int = 64
  val numTrees: Int = 100
  val maxIter: Int = 150
  val stepSize: Double = 0.03
  val seed: Long = 42L
  val webPort: Int = 5000

  def ensureDirectories(): Unit =
    Seq(parsingOutputPath, modelPath, outputPath, logPath)
      .foreach(path => Files.createDirectories(path))
}
