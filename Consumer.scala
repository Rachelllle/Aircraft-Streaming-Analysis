import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.streaming.Trigger
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._

// test first consumer
object Consumer {
  def main(args: Array[String]): Unit = {
    
    // Initializeation session
    val spark = SparkSession.builder()
      .appName("AircraftImageConsumer")
      .master("local[*]")
      .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    import spark.implicits._
    //val sharedSourcePath = "data/output"

    // defini le schema pour lire les binary files
    val binaryFileSchema = StructType(Seq(
      StructField("path", StringType, nullable = false),
      StructField("modificationTime", TimestampType, nullable = false),
      StructField("length", LongType, nullable = false),
      StructField("content", BinaryType, nullable = true)
    ))

    val strImages = spark.readStream
      .format("binaryFile")
      .schema(binaryFileSchema)
      .option("pathGlobFilter", "*.jpg") // files d type image jpg
      .load("data/output")

    val processedAircraftDF = strImages
      .withColumn("fileName", element_at(split($"path", "/"), -1))
      .withColumn("aircraftType", split($"fileName", "_")(0))

    // nbr of  aircraft type
    val aircraftCounts = processedAircraftDF.groupBy("aircraftType").count()


    val query = aircraftCounts.writeStream
      .outputMode("complete") 
      .format("console") // f tab
      .option("checkpointLocation", "data/checkpointconsumr")
      .trigger(Trigger.ProcessingTime("20 seconds"))
      .start() /

    query.awaitTermination() 
  }
}