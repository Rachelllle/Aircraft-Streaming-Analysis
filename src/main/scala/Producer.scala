import org.apache.spark.{SparkConf, SparkContext}
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.hadoop.io.IOUtils
import org.apache.spark.util.SerializableConfiguration

object Producer {
  def main(args: Array[String]): Unit = {
    val inputPath  = "data/input"
    val outputPath = "data/output"
    val nbPhotos   = 20

    val conf = new SparkConf()
      .setAppName("ImageProducer")
      .setMaster("local[*]")
    val sc = new SparkContext(conf)
    sc.setLogLevel("WARN")

    val confSer = new SerializableConfiguration(sc.hadoopConfiguration)

    val sourcePaths = sc.binaryFiles(inputPath).keys.toLocalIterator

    val groups = sourcePaths.grouped(nbPhotos)

    var batchId = 0
    while (groups.hasNext) {
      val batch = groups.next().toSeq

      sc.parallelize(batch, numSlices = batch.length).foreachPartition { partition =>
        val fs = FileSystem.get(confSer.value)
        partition.foreach { srcUri =>
          val srcPath  = new Path(srcUri)
          val fileName = srcPath.getName
          val outFile  = new Path(outputPath, fileName)
          val in  = fs.open(srcPath)
          val out = fs.create(outFile, true)
          try {
            IOUtils.copyBytes(in, out, confSer.value, false)
          } finally {
            in.close()
            out.close()
          }
        }
      }

      println(s"Batch $batchId : ${batch.length} images écrites dans $outputPath")
      batchId += 1
    }

    println("Traitement terminé. Tous les fichiers ont été traités")
    sc.stop()
  }
}