ThisBuild / scalaVersion := "2.12.18"
ThisBuild / version := "0.1.0"
ThisBuild / organization := "com.aircraft"

lazy val root = (project in file("."))
  .settings(
    name := "Aircraft-Analysis-Engine_scala",
    libraryDependencies ++= Seq(
      "org.apache.spark" %% "spark-sql" % "3.5.1",
      "org.apache.spark" %% "spark-mllib" % "3.5.1",
      "com.lihaoyi" %% "ujson" % "3.3.1",
      "com.sparkjava" % "spark-core" % "2.9.4",
      "org.slf4j" % "slf4j-simple" % "2.0.13"
    ),
    run / fork := false,
    javacOptions ++= Seq("-source", "11", "-target", "11"),
    Compile / mainClass := Some("com.aircraft.analysis.Main"),
    javaOptions ++= Seq(
      "--add-opens=java.base/java.lang=ALL-UNNAMED",
      "--add-opens=java.base/java.lang.invoke=ALL-UNNAMED",
      "--add-opens=java.base/java.lang.reflect=ALL-UNNAMED",
      "--add-opens=java.base/java.io=ALL-UNNAMED",
      "--add-opens=java.base/java.net=ALL-UNNAMED",
      "--add-opens=java.base/java.nio=ALL-UNNAMED",
      "--add-opens=java.base/java.util=ALL-UNNAMED",
      "--add-opens=java.base/java.util.concurrent=ALL-UNNAMED",
      "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED",
      "--add-opens=java.base/sun.security.action=ALL-UNNAMED"
    )
  )
