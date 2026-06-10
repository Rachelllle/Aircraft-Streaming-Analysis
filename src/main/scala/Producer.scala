import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.streaming.Trigger
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.util.SerializableConfiguration
import org.apache.spark.sql.types._


object Producer {
  def main(args: Array[String]): Unit = {
    val inputPath      = "data/input"
    val outputPath     = "data/output"
    val nbPhotos       = 20      // nb  photos par batch
    val temps          = 5       // cadence en sec

    val spark = SparkSession.builder()
      .appName("ImageProducer")
      .master("local[*]")
      .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    val confSer = new SerializableConfiguration(spark.sparkContext.hadoopConfiguration)

    val binaryFileSchema = StructType(Seq(
      StructField("path", StringType, nullable = false),
      StructField("modificationTime", TimestampType, nullable = false),
      StructField("length", LongType, nullable = false),
      StructField("content", BinaryType, nullable = true)
    ))

    val images = spark.readStream
      .format("binaryFile")
      .schema(binaryFileSchema)
      .option("pathGlobFilter", "*.jpg")
      .option("maxFilesPerTrigger", nbPhotos)
      .load(inputPath)

    val query = images.writeStream
      .foreachBatch { (batchDF: org.apache.spark.sql.DataFrame, batchId: Long) =>
        batchDF.select("path", "content").foreachPartition { rows: Iterator[org.apache.spark.sql.Row] =>
          val fs = FileSystem.get(confSer.value)
          rows.foreach { row =>
            val srcPath  = row.getAs[String]("path")
            val content  = row.getAs[Array[Byte]]("content")
            val fileName = new Path(srcPath).getName
            val outFile  = new Path(outputPath, fileName)
            val out = fs.create(outFile, true)
            try {
              out.write(content)
            } finally {
              out.close()
            }
          }
        }
        println(s"Batch $batchId : ${batchDF.count()} images écrites dans $outputPath")
      }
      .option("checkpointLocation", "data/checkpoint")
      .trigger(Trigger.ProcessingTime(s"$temps seconds"))
      .start()

    query.awaitTermination()
  }
}