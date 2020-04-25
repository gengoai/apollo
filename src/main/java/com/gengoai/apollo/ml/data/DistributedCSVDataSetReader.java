/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package com.gengoai.apollo.ml.data;

import com.gengoai.apollo.ml.DataSet;
import com.gengoai.apollo.ml.Datum;
import com.gengoai.apollo.ml.StreamingDataSet;
import com.gengoai.apollo.ml.observation.Variable;
import com.gengoai.io.CSV;
import com.gengoai.io.resource.Resource;
import com.gengoai.stream.StreamingContext;
import com.gengoai.stream.spark.SparkStream;
import lombok.NonNull;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SQLContext;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

/**
 * <p>
 * A {@link DataSetReader} for delimiter separated files that generates a distributed {@link DataSet}. Rows in the file
 * represents {@link Datum} and columns {@link com.gengoai.apollo.ml.observation.Observation}s. Generated {@link Datum}
 * have an observation per column in the file  with the source names equalling the column names. The header must be
 * specified in the file or through the CSV object passed into the constructor. Additionally, a schema must be specified
 * via Map of source (column) names to {@link ColumnConverter}.
 * </p>
 *
 * @author David B. Bracewell
 */
public class DistributedCSVDataSetReader implements DataSetReader {
   private static final long serialVersionUID = 1L;
   private final CSV csv;
   private final Map<String, ColumnConverter> columnToVariable;

   /**
    * Instantiates a new DistributedCSVDataSetReader
    *
    * @param csv              the definition of the csv file
    * @param columnToVariable the schema
    */
   public DistributedCSVDataSetReader(@NonNull CSV csv,
                                      @NonNull Map<String, ColumnConverter> columnToVariable) {
      this.csv = csv;
      this.columnToVariable = columnToVariable;
      if(!csv.getHasHeader() && csv.getHeader().isEmpty()) {
         throw new IllegalArgumentException("Either the CSV must have a header or one must be defined.");
      }
   }

   @Override
   public DataSet read(@NonNull Resource dataResource) throws IOException {
      SQLContext sqlContext = new SQLContext(StreamingContext.distributed().sparkSession());
      org.apache.spark.sql.Dataset<Row> rows = sqlContext.read()
                                                         .option("delimiter", csv.getDelimiter())
                                                         .option("escape", csv.getEscape())
                                                         .option("quote", csv.getQuote())
                                                         .option("comment", csv.getComment())
                                                         .option("header", csv.getHasHeader())
                                                         .csv(dataResource.path());
      List<String> headers = Arrays.asList(rows.columns());
      DataSet dataSet = new StreamingDataSet(new SparkStream<>(rows.toJavaRDD()
                                                                   .map(row -> rowToDatum(headers, row))));
      for(String column : headers) {
         dataSet.updateMetadata(column, m -> m.setType(Variable.class));
      }
      return dataSet;
   }

   private Datum rowToDatum(List<String> headers, Row row) {
      Datum datum = new Datum();
      for(int i = 0; i < headers.size(); i++) {
         String column = headers.get(i);
         datum.put(column,
                   columnToVariable.get(column)
                                   .convert(column,
                                            row.getString(
                                                  i)));
      }
      return datum;
   }
}//END OF DistributedCSVDataSetReader
