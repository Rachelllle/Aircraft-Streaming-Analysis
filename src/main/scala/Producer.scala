import org.apache.spark.{SparkConf, SparkContext}
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.hadoop.io.IOUtils
import org.apache.spark.util.SerializableConfiguration

object Producer {
  def main(args: Array[String]): Unit = {
    val inputPath      = "data/input"
    val outputPath     = "data/output"
    val nbPhotos       = 20      // nb photos par batch
    val batchDuration  = 5       // cadence en sec
    val isLoop         = true

    val conf = new SparkConf()
      .setAppName("ImageProducer")
      .setMaster("local[*]")
    val sc = new SparkContext(conf)
    sc.setLogLevel("WARN")

    val confSer = new SerializableConfiguration(sc.hadoopConfiguration)

    while (isLoop) {
      val rdd = sc.binaryFiles(inputPath)

      rdd.coalesce(1).foreachPartition { partitionIterator =>
        val fs = FileSystem.get(confSer.value)

        partitionIterator.grouped(nbPhotos).foreach { batch =>
          batch.foreach { case (path, content) =>
            val srcPath  = new Path(path)
            val fileName = srcPath.getName
            val outFile  = new Path(outputPath, fileName)
            val in  = content.open()
            val out = fs.create(outFile, true)
            try {
              IOUtils.copyBytes(in, out, confSer.value, false)
            } finally {
              in.close()
              out.close()
            }
          }
          println(s"Batch traité : ${batch.length} images écrites dans $outputPath")
        }
      }

      Thread.sleep(batchDuration * 1000)
    }

    sc.stop()
  }
}