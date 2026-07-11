import java.io.FileInputStream
import java.util.Properties

case class AppConfig(
  mode: String,
  inputPath: String,
  intermediatePath: String,
  outputPath: String,
  checkpointPath: String,
  modelPath: String,
  batchSize: Int,
  streamInterval: Int,
  imageWidth: Int,
  imageHeight: Int
)

object AppConfig {
  def load(path: String = "config/application.properties"): AppConfig = {
    val props = new Properties()
    val in = new FileInputStream(path)
    try props.load(in) finally in.close()

    AppConfig(
      mode             = props.getProperty("app.mode", "train"),
      inputPath        = props.getProperty("input.path"),
      intermediatePath = props.getProperty("intermediate.path"),
      outputPath       = props.getProperty("output.path"),
      checkpointPath   = props.getProperty("checkpoint.path"),
      modelPath        = props.getProperty("model.path"),
      batchSize        = props.getProperty("batch.size").toInt,
      streamInterval   = props.getProperty("stream.interval").toInt,
      imageWidth       = props.getProperty("image.width").toInt,
      imageHeight      = props.getProperty("image.height").toInt
    )
  }
}
