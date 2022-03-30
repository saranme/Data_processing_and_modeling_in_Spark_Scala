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
