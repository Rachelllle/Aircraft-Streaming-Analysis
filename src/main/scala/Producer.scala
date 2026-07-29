import org.apache.spark.{SparkConf, SparkContext}
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.hadoop.io.IOUtils
import org.apache.spark.util.SerializableConfiguration

object Producer {
  def main(args: Array[String]): Unit = {
    val inputPath  = "data/input"
    val outputPath = "data/output"
    val nbPhotos   = 20      // taille du batch
    val interval   = 3       // délai entre chaque batch
    val recursive  = true    // lecture des sous dossiers ?

    val conf = new SparkConf()
      .setAppName("ImageProducer")
      .setMaster("local[*]")
    val sc = new SparkContext(conf)
    sc.setLogLevel("WARN")

    if (recursive) {
      sc.hadoopConfiguration.setBoolean("mapreduce.input.fileinputformat.input.dir.recursive", true)
    }

    val confSer = new SerializableConfiguration(sc.hadoopConfiguration)
    val fs = FileSystem.get(sc.hadoopConfiguration)


    val sourcePaths = sc.binaryFiles(inputPath).keys.toLocalIterator
    val groups = sourcePaths.grouped(nbPhotos)

    val tempsDebut = System.currentTimeMillis()
    var totalImages = 0
    var batchId = 0
    var tempsTraitementPur = 0.0

    while (groups.hasNext) {
      val batch = groups.next().toSeq
      val tempsBatchDebut = System.currentTimeMillis()


      batch.foreach { srcUri =>
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

      val dureeBatch = (System.currentTimeMillis() - tempsBatchDebut) / 1000.0
      tempsTraitementPur += dureeBatch
      totalImages += batch.length
      println(f"Batch $batchId : ${batch.length} images en $dureeBatch%.2f s")
      batchId += 1

      if (groups.hasNext) {
        Thread.sleep(interval * 1000)
      }
    }

    val dureeTotale = (System.currentTimeMillis() - tempsDebut) / 1000.0
    val debitTotal = if (dureeTotale > 0) totalImages / dureeTotale else 0.0
    val debitPur   = if (tempsTraitementPur > 0) totalImages / tempsTraitementPur else 0.0

    println("********* RÉSUMÉ *********")
    println(f"$totalImages images traitées")
    println(f"Temps total (avec pauses)  : $dureeTotale%.2f s  -> débit cadencé : $debitTotal%.2f images/s")
    println(f"Temps de traitement pur    : $tempsTraitementPur%.2f s  -> débit réel machine : $debitPur%.2f images/s")

    sc.stop()
  }
}