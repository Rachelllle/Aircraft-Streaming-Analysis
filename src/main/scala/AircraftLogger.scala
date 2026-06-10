package com.aircraft.analysis

import java.nio.file.Files
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter
import java.util.logging.{FileHandler, Formatter, Level, LogRecord, Logger, SimpleFormatter}

object AircraftLogger {
  val logger: Logger = {
    Config.ensureDirectories()

    val logger = Logger.getLogger("AircraftAnalysisScala")
    logger.setUseParentHandlers(true)
    logger.setLevel(Level.INFO)

    if (logger.getHandlers.isEmpty) {
      val timestamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd_HH-mm-ss"))
      val logFile = Config.logPath.resolve(s"session_$timestamp.log")
      Files.createDirectories(logFile.getParent)

      val fileHandler = new FileHandler(logFile.toString)
      val formatter: Formatter = new SimpleFormatter {
        override def format(record: LogRecord): String =
          s"${record.getMillis} | ${record.getLoggerName} | ${record.getLevel} | ${record.getMessage}%n"
      }

      fileHandler.setFormatter(formatter)
      logger.addHandler(fileHandler)
    }

    logger
  }
}
