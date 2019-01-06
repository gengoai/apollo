/*
 * (c) 2005 David B. Bracewell
 *
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
 *
 */

package com.gengoai.apollo.ml.data.format;

import com.gengoai.apollo.ml.Example;
import com.gengoai.apollo.ml.Feature;
import com.gengoai.apollo.ml.Instance;
import com.gengoai.io.CSV;
import com.gengoai.io.resource.Resource;
import com.gengoai.stream.MStream;
import com.gengoai.stream.SparkStream;
import com.gengoai.stream.StreamingContext;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SQLContext;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * <p>
 * Distributed version of {@link CSVDataFormat} using Spark to parse the files. Converts delimited separated values
 * (e.g. csv and tsv) files into examples. Rows in the file represent individual examples and columns represent the
 * features and label. The label column is specified using the column name or column index. For unlabelled data, set the
 * column index to -1, by default is set to the first column (i.e. column 0).
 * </p>
 * <p>
 * If file does not have a header and one is not provided in CSV object passed to the constructor, column names will be
 * automatically created as AutoColumn_INDEX.
 * </p>
 *
 * @author David B. Bracewell
 */
public class DistributedCSVDataFormat extends CSVDataFormat {
   private static final long serialVersionUID = 1L;

   /**
    * Instantiates a new Distributed csv data format.
    *
    * @param csv the csv
    */
   public DistributedCSVDataFormat(CSV csv) {
      super(csv);
   }

   /**
    * Instantiates a new Distributed csv data format.
    *
    * @param csv        the csv
    * @param labelIndex the label index
    */
   public DistributedCSVDataFormat(CSV csv, int labelIndex) {
      super(csv, labelIndex);
   }

   /**
    * Instantiates a new Distributed csv data format.
    *
    * @param csv         the csv
    * @param labelColumn the label column
    */
   public DistributedCSVDataFormat(CSV csv, String labelColumn) {
      super(csv, labelColumn);
   }

   @Override
   public MStream<Example> read(Resource location) throws IOException {
      SQLContext sqlContext = new SQLContext(StreamingContext.distributed().sparkSession());
      org.apache.spark.sql.Dataset<Row> rows = sqlContext.read()
                                                         .option("delimiter", csv.getDelimiter())
                                                         .option("escape", csv.getEscape())
                                                         .option("quote", csv.getQuote())
                                                         .option("comment", csv.getComment())
                                                         .option("header", csv.getHasHeader())
                                                         .csv(location.path());
      List<String> headers = Arrays.asList(rows.columns());
      final int li = labelColumnIndex >= 0 ? labelColumnIndex : headers.indexOf(labelColumn);
      return new SparkStream<>(rows.toJavaRDD()
                                   .map(row -> {
                                      List<Feature> features = new ArrayList<>();
                                      String label = li >= 0 ? row.getString(li) : null;
                                      for (int i = 0; i < row.size(); i++) {
                                         if (i != li) {
                                            while (headers.size() < i) {
                                               headers.add("AutoColumn_" + i);
                                            }
                                            Feature feature = createFeature(headers.get(i), row.getString(i));
                                            if (feature != null) {
                                               features.add(feature);
                                            }
                                         }
                                      }
                                      return new Instance(label, features);
                                   }));
   }
}//END OF DistributedCSVDataSource
