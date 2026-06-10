import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.streaming.Trigger
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._

// Consumer.scala THIS IS A FIRST VERSION MAY NEED ADJUSTEMENTS BLAAAAAAAAABLA
object Consumer {
  def main(args: Array[String]): Unit = {
    
    // 1. Initializeation
    val spark = SparkSession.builder()
      .appName("AircraftImageConsumer")
      .master("local[*]") // Run locally using all available cores [cite: 189]
      .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    import spark.implicits._ // Enablers handy implicits (As shown on Page 63) [cite: 755]

    val sharedSourcePath = "data/output" 

    // 2. Define the schema to read binary files
    val binaryFileSchema = StructType(Seq(
      StructField("path", StringType, nullable = false),
      StructField("modificationTime", TimestampType, nullable = false),
      StructField("length", LongType, nullable = false),
      StructField("content", BinaryType, nullable = true)
    ))
    // Structured Streaming treats this directory as an Unbounded Table
    val streamingImagesDF = spark.readStream
      .format("binaryFile")
      .schema(binaryFileSchema)
      .option("pathGlobFilter", "*.jpg") // Focus only on image files
      .load("data/output")

    //.xtract Aircraft Type from the filename
    // Let's assume your colleague saves images with names like "Boeing747_id123.jpg" or "AirbusA320_99.jpg"

    val processedAircraftDF = streamingImagesDF
      .withColumn("fileName", element_at(split($"path", "/"), -1))
      .withColumn("aircraftType", split($"fileName", "_")(0))

    // CONSOLIDATION: Running total of aircraft instances (Unbounded table principle) [
    val globalAircraftCounts = processedAircraftDF.groupBy("aircraftType").count()


    val query = globalAircraftCounts.writeStream
      .outputMode("complete") 
      .format("console")      // Prints cleanly as a structured table in your terminal logs [cite: 782, 786]
      .option("checkpointLocation", "data/checkpoint_consumer")
      .trigger(Trigger.ProcessingTime("20 seconds"))              // Fires the trigger every 20 seconds
      .start() /

    query.awaitTermination() 
  }
}