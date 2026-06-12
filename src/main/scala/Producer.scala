import org.apache.spark.{SparkConf, SparkContext}
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.hadoop.io.IOUtils
import org.apache.spark.util.SerializableConfiguration


object Producer {
  def main(args: Array[String]): Unit = {
    val inputPath      = "data/input"
    val outputPath     = "data/output"
    val nbPhotos       = 20      // nb photos par batch

    // Option -> copy par défaut si non move
    val mode = if (args.length > 0) args(0) else "copy"
    val moveFiles = mode.equalsIgnoreCase("move")

    val conf = new SparkConf()
      .setAppName("ImageProducer")
      .setMaster("local[*]")
    val sc = new SparkContext(conf)
    sc.setLogLevel("WARN")

    val rdd = sc.binaryFiles(inputPath)
    val batch = rdd.take(nbPhotos)

    val confSer = new SerializableConfiguration(sc.hadoopConfiguration)

    batch.foreach { case (path, content) =>
      val fs = FileSystem.get(confSer.value)
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

      if (moveFiles) {
        fs.delete(srcPath, false)
      }
    }

    val action = if (moveFiles) "Déplacement" else "Copie"
    println(s"$action terminé.. vers : $outputPath (${batch.length} images)")
    sc.stop()
  }
}