'''
Hola Sara,
dado que lo has subido a una URL, vamos a utilizarla. Desde dentro del contenedor deberías ejecutar:
wget https://raw.githubusercontent.com/saranme/x/main/rain_tomorrow_in_australia_mod.csv -P /root/wget https://raw.githubusercontent.com/saranme/x/main/rain_tomorrow_in_australia_mod.csv

'''
// abrir el docker desktop. En terminal:
// docker system prune
// docker run -dit -p 4040:4040 -p 9070:50070 -p 8080:8080 -p 8081:8081 --name bigdata mafernandezd/big_data:v4
// docker exec -it bigdata bash
// cd /root/spark-2.4.4-bin-hadoop2.7/
// bin/spark-shell
// chuleta: https://learn-eu-central-1-prod-fleet01-xythos.content.blackboardcdn.com/5d3210a6eb3d6/2089220?X-Blackboard-Expiration=1647648000000&X-Blackboard-Signature=WxJOXTXU5j7JQBFYcj5QVrAi6tREVV6epnnjikEfsPg%3D&X-Blackboard-Client-Id=347294&response-cache-control=private%2C%20max-age%3D21600&response-content-disposition=inline%3B%20filename%2A%3DUTF-8%27%27Docker%2520comandos.txt&response-content-type=text%2Fplain&X-Amz-Security-Token=IQoJb3JpZ2luX2VjEEoaDGV1LWNlbnRyYWwtMSJHMEUCIFsvVnbVLYPn1QHeKYyL%2FN6tsgvHMF6jH8eAALfT5TP2AiEAs0M6u9AUyVAT6TI1BWUHavKWLCyfUAzu57NGqo3WuJkqiQQIw%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARACGgw2MzU1Njc5MjQxODMiDB7IulETRLK2J%2FK2PyrdA%2B7vgY6Rw1NaAZa%2BKqprcYAhB2wa%2B9hwT3ZAgt4h3zcSsD%2FKRfvzT2FxOn%2B7OFc2TTKKC24k7MNYHvk05UKoo7Zd2gfGV40M5KB%2BmCpaaqBB%2FVzIksxEylwAHtN9xwgZO%2FSsZ4RFh1Miz1XodwJCKWgp%2FE0KbHs7zTEGjxPM55YgCJ%2FhKl1kUhXwTefbAI7tbOOYANMAOxu11y%2BSVOs3pMIaVpGUgTaAKcK7vDj8fTuhZKcNTF1qPPQc2tigoJaIhQuoQvnH1m%2FC4x%2FxMbvYUhP3E8pPdolmV2Rtcf8ksd52luFmlcjfYzsdStZ75bu6Imwt7iKEWY%2BWruPsWt3vrj0wQERipzd4OrimeDXBObgH8JqscZeXAw7jv%2BmRzDtRhpRhBfRSK%2FI69WoP49VFaRc7m98KxQlNBbCleRTSzHe0LILA98qNq5QuW%2BIb5eGV94PHutxd9VlN%2FSG6qjniQm%2ByhAt8Yb5JFFG%2B7tpICneVqm4nSUnM8AqzXmLxJvDbYO4cRaGg7phlRZQUb1YbVZAdlITL%2FykOSSAhQJKcJ7rr%2FlAZue%2Bgz8wnZnVj3OadCAKN1sF%2BfqMY%2B5ZjwJ27I7Vj1u%2BKlMdmQ4SynDhUssjmxZ4%2FueMRa9hBgfykDDDXl9ORBjqlAasVc5C78lqGtZvGlEKpnjPnxGMNp2swMk76SCOMb3uZlZ9FmasTmrxvryZviDXvtlnnHu6RCnt%2B8qhwEjbmRdFWk3n2e4Fpx%2BLQwfXDmZlWtWCy%2BltlR9LelXNJkqXjqT8eDS3NYCLLnXXeIkdQgsJ5fauyFberT%2BqwmikV%2FuS3rQacdsBAAjeAh26t8GHkmjj4wq6bx6EsvmD%2F405RIbPLwLMATg%3D%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20220318T180000Z&X-Amz-SignedHeaders=host&X-Amz-Expires=21600&X-Amz-Credential=ASIAZH6WM4PL4KL2PSV6%2F20220318%2Feu-central-1%2Fs3%2Faws4_request&X-Amz-Signature=8b99da54b772a8d938973b15f0e13e2acc0ed436e08e052a3917ae9b4cbeb6e1

import org.apache.spark.sql.{Dataset, Row}
import org.apache.spark.sql.types.{DoubleType, StringType}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{StringIndexer, StandardScaler, VectorAssembler}


def model_rain(train_df: Dataset[Row], test_df: Dataset[Row]): Double = {
	// división de variables según tipo
	val categoricals = Array("WindGustDir", "WindDir9am", "WindDir3pm", "RainToday", "RainTomorrow")
	val numericals = Array("MinTemp","MaxTemp","Rainfall","Evaporation","Sunshine","WindGustSpeed","WindSpeed9am","WindSpeed3pm","Humidity9am","Humidity3pm","Pressure9am","Pressure3pm","Cloud9am","Cloud3pm","Temp9am","Temp3pm","PXW")

	// Indexación de variables categóricas
	val indexers = (categoricals ++ Array("target")).map { col_name =>
	               new StringIndexer().setInputCol(col_name).setOutputCol(col_name + "Index")}

	val categories_indexed = categoricals.map(col_name => col_name + "Index")

	// Unión de las variables en un vector
	val assembler = new VectorAssembler().setInputCols(categories_indexed ++ numericals).setOutputCol("features")

	// Normalización de variables
	val scaler = new StandardScaler().setInputCol("features").setOutputCol("scaled_features").setWithStd(true).setWithMean(true)

	// Modelo de regresión logística
	val lr = new LogisticRegression().setMaxIter(10).setLabelCol("targetIndex")

	// Creación de pipeline
	val pipeline = new Pipeline().setStages(indexers ++ Array(assembler, scaler, lr))

	// Entrenamiento y evaluación del modelo
	val model = pipeline.fit(train_df)

	val prediccion = model.transform(test_df)
	//val evaluator = new BinaryClassificationEvaluator().setLabelCol("targetIndex").setRawPredictionCol("rawPrediction")
	//val test_auc = evaluator.evaluate(prediccion)
	//println("Precisión del modelo: " + test_auc)

	// Creación de columna para cada elemento del vector
	val vecToArray = udf((xs: Vector) => xs.toArray)
	// Probabilidad de lluvia
	val df_lluvia = prediccion.withColumn("prob_array" , vecToArray(col("probability")))
	val elements = Array("No", "Sí")
	val sqlExpr = elements.zipWithIndex.map{case (alias, idx) => col("prob_array").getItem(idx).as(alias)}
	val prediccion_lluvia = df_lluvia.select(sqlExpr : _*)
	prediccion_lluvia.select("Sí").take(1)(0).get(0).asInstanceOf[Double]
}

def rain_prob(city: String = null): Double = {
	// división de variables numéricas
	val numericals = Array("MinTemp","MaxTemp","Rainfall","Evaporation","Sunshine","WindGustSpeed","WindSpeed9am","WindSpeed3pm","Humidity9am","Humidity3pm","Pressure9am","Pressure3pm","Cloud9am","Cloud3pm","Temp9am","Temp3pm","PXW")

	// Creación de dataframe
	val df = spark.read.format("csv").option("header", "true").option("delimiter", ",").option("inferSchema", "true").load("rain_tomorrow_in_australia_mod.csv")
	// Eliminación duplicados
	val df2 = df.dropDuplicates()

	// Reemplazo de nulos
	val df3 = numericals.foldLeft(
		df2) {(aux_df, col_name) => aux_df.withColumn(
			col_name, aux_df.col(col_name).cast(DoubleType)
		)}

	val df4 = df3
	.withColumn("MinTemp", coalesce(col("MinTemp"), lit(df3.agg(mean("MinTemp").alias("MinTemp")).select("MinTemp").take(1)(0).get(0).asInstanceOf[Double])))
	.withColumn("MaxTemp", coalesce(col("MaxTemp"), lit(df3.agg(mean("MaxTemp").alias("MaxTemp")).select("MaxTemp").take(1)(0).get(0).asInstanceOf[Double])))
	.withColumn("Rainfall", coalesce(col("Rainfall"), lit(df3.agg(mean("Rainfall").alias("Rainfall")).select("Rainfall").take(1)(0).get(0).asInstanceOf[Double])))
	.withColumn("Evaporation", coalesce(col("Evaporation"), lit(df3.agg(mean("Evaporation").alias("Evaporation")).select("Evaporation").take(1)(0).get(0).asInstanceOf[Double])))
	.withColumn("Sunshine", coalesce(col("Sunshine"), lit(df3.agg(mean("Sunshine").alias("Sunshine")).select("Sunshine").take(1)(0).get(0).asInstanceOf[Double])))
	.withColumn("WindGustSpeed", coalesce(col("WindGustSpeed"), lit(df3.agg(mean("WindGustSpeed").alias("WindGustSpeed")).select("WindGustSpeed").take(1)(0).get(0).asInstanceOf[Double])))
	.withColumn("WindSpeed9am", coalesce(col("WindSpeed9am"), lit(df3.agg(mean("WindSpeed9am").alias("WindSpeed9am")).select("WindSpeed9am").take(1)(0).get(0).asInstanceOf[Double])))
	.withColumn("WindSpeed3pm", coalesce(col("WindSpeed3pm"), lit(df3.agg(mean("WindSpeed3pm").alias("WindSpeed3pm")).select("WindSpeed3pm").take(1)(0).get(0).asInstanceOf[Double])))
	.withColumn("Humidity9am", coalesce(col("Humidity9am"), lit(df3.agg(mean("Humidity9am").alias("Humidity9am")).select("Humidity9am").take(1)(0).get(0).asInstanceOf[Double])))
	.withColumn("Humidity3pm", coalesce(col("Humidity3pm"), lit(df3.agg(mean("Humidity3pm").alias("Humidity3pm")).select("Humidity3pm").take(1)(0).get(0).asInstanceOf[Double])))
	.withColumn("Pressure9am", coalesce(col("Pressure9am"), lit(df3.agg(mean("Pressure9am").alias("Pressure9am")).select("Pressure9am").take(1)(0).get(0).asInstanceOf[Double])))
	.withColumn("Pressure3pm", coalesce(col("Pressure3pm"), lit(df3.agg(mean("Pressure3pm").alias("Pressure3pm")).select("Pressure3pm").take(1)(0).get(0).asInstanceOf[Double])))
	.withColumn("Cloud9am", coalesce(col("Cloud9am"), lit(df3.agg(mean("Cloud9am").alias("Cloud9am")).select("Cloud9am").take(1)(0).get(0).asInstanceOf[Double])))
	.withColumn("Cloud3pm", coalesce(col("Cloud3pm"), lit(df3.agg(mean("Cloud3pm").alias("Cloud3pm")).select("Cloud3pm").take(1)(0).get(0).asInstanceOf[Double])))
	.withColumn("Temp9am", coalesce(col("Temp9am"), lit(df3.agg(mean("Temp9am").alias("Temp9am")).select("Temp9am").take(1)(0).get(0).asInstanceOf[Double])))
	.withColumn("Temp3pm", coalesce(col("Temp3pm"), lit(df3.agg(mean("Temp3pm").alias("Temp3pm")).select("Temp3pm").take(1)(0).get(0).asInstanceOf[Double])))
	.withColumn("WindGustDir", coalesce(col("WindGustDir"), lit("unknown")))
	.withColumn("WindDir9am", coalesce(col("WindDir9am"), lit("unknown")))
	.withColumn("WindDir3pm", coalesce(col("WindDir3pm"), lit("unknown")))
	.withColumn("RainToday", coalesce(col("RainToday"), lit("unknown")))
	.withColumn("target", coalesce(col("RainTomorrow"), lit("unknown")))
	
	// Filtrar por ciudad o país
	var loc_values = List[String]()
	if (df4.select("Location").distinct().collect().map(_.mkString).toList.contains(city)) {
		loc_values = df4.filter(col("Location") === city).select("Location").distinct().collect().map(_.mkString).toList
	} 
	else {
		loc_values = df4.select("Location").distinct().collect().map(_.mkString).toList
	}

	val df5 = df4.filter($"Location".isin(loc_values:_*))

	// Eliminación de columnas pedidas en el enunciado de la práctica
	val df_ = df5.drop("RISK_MM", "Date", "Location")

	// Partición de dataframe en train y test
	val Array(train_df, test_df) = df_.randomSplit(Array(0.8, 0.2))

	// Aplicación de modelo logarítimico y predicción de lluvia
	model_rain(train_df, test_df)
}

println("La probabilidad de que llueva en Canberra es de %1.2f.".format(rain_prob("Canberra")))
println("La probabilidad de que llueva en Australia es de %1.2f.".format(rain_prob()))
println("La probabilidad de que llueva en Albury es de %1.2f.".format(rain_prob("Albury")))
