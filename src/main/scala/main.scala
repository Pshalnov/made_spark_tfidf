package tdifd

import org.apache.spark.sql._
import org.apache.spark.sql.functions._

object main {
  def main(args: Array[String]): Unit = {
    // Создает сессию спарка
    val spark = SparkSession.builder()
      // адрес мастера
      .master("local[*]")
      // имя приложения в интерфейсе спарка
      .appName("made-tfidf")
      // взять текущий или создать новый
      .getOrCreate()

    // синтаксический сахар для удобной работы со спарк
    import spark.implicits._

    // прочитаем датасет https://www.kaggle.com/andrewmvd/trip-advisor-hotel-reviews

    val df = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv("/Users/pavel/IdeaProjects/untitled/tripadvisor_hotel_reviews.csv")
      .withColumn("id", monotonically_increasing_id())

    // Приводим все к нижнему регистру
    val Lower = df
      .withColumn("ReviewLower", lower(col("Review")))

    // Удаляем все спец символы
    val LowerDelSumbols = Lower
      .withColumn("ReviewLowerClean",
        split(regexp_replace(col("ReviewLower"), "[^0-9a-z ]", ""), " "))


    val SentLength = LowerDelSumbols
      .withColumn("SentLen", size(col("ReviewLowerClean")))

    val Exploded = SentLength
      .withColumn("Word", explode(col("ReviewLowerClean")))
      .filter(col("Word").notEqual(""))

    // Считаем частоту слов
    val CountWord = Exploded
      .groupBy("id", "Word")
      .agg(
        count("Word") as "CountWord",
        first("Sentlen") as "SentLen"
      )

    val Tf = CountWord
      .withColumn("idTF", col("CountWord") / col("SentLen"))

    // Посчитать количество документов со словом
    val CountDoc = Exploded
      .groupBy("Word")
      .agg(
        countDistinct("id") as "CountDoc"
      )

    // Берем 100 самых встречаемых
    val DocBest = CountDoc
      .orderBy(desc("CountDoc"))
      .limit(100)

    val dfLen = df.count().toDouble

    val resultIdf = udf((value: Int) => math.log(dfLen/value.toDouble))

    val Idf = DocBest
      .withColumn("idIDF", resultIdf(col("CountDoc")))

    // Джойним
    val Joined = Tf
      .join(Idf, Seq("Word"), "inner")
      .withColumn("TFIDF", col("idTF") * col("idIDF"))

    val Pivot = Joined
      .groupBy("id")
      .pivot(col("Word"))
      .agg(
        first(col("TFIDF"))
      )
      .na.fill(0.0)

    Pivot.show()
    Pivot
      .coalesce(1)
      .write
      .option("sep", ",")
      .option("header", "true")
  }
}