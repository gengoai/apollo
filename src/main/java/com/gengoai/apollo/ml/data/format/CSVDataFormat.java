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
import com.gengoai.io.CSVReader;
import com.gengoai.io.resource.Resource;
import com.gengoai.math.Math2;
import com.gengoai.stream.MStream;
import com.gengoai.stream.StreamingContext;
import com.gengoai.string.Strings;

import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

/**
 * <p>
 * Converts delimited separated values (e.g. csv and tsv) files into examples. Rows in the file represent individual
 * examples and columns represent the features and label. The label column is specified using the column name or column
 * index. For unlabelled data, set the column index to -1, by default is set to the first column (i.e. column 0).
 * </p>
 * <p>
 * If file does not have a header and one is not provided in CSV object passed to the constructor, column names will be
 * automatically created as AutoColumn_INDEX.
 * </p>
 *
 * @author David B. Bracewell
 */
public class CSVDataFormat implements DataFormat, Serializable {
   private static final long serialVersionUID = 1L;
   /**
    * The Csv.
    */
   protected final CSV csv;
   /**
    * The Label column.
    */
   protected String labelColumn;
   /**
    * The Label column index.
    */
   protected int labelColumnIndex;

   /**
    * Instantiates a new CSV data source with label index set to the first column.
    *
    * @param csv the csv parameters
    */
   public CSVDataFormat(CSV csv) {
      this(csv, 0);
   }

   /**
    * Instantiates a new Csv data source.
    *
    * @param csv        the csv parameters
    * @param labelIndex the label index
    */
   public CSVDataFormat(CSV csv, int labelIndex) {
      this.csv = csv;
      this.labelColumn = null;
      this.labelColumnIndex = labelIndex;
   }

   /**
    * Instantiates a new Csv data source.
    *
    * @param csv         the csv parameters
    * @param labelColumn the label column
    */
   public CSVDataFormat(CSV csv, String labelColumn) {
      this.csv = csv;
      this.labelColumn = labelColumn;
      this.labelColumnIndex = -1;
   }


   /**
    * Create feature feature.
    *
    * @param featureName the feature name
    * @param columnValue the column value
    * @return the feature
    */
   protected Feature createFeature(String featureName, String columnValue) {
      if (columnValue.equals("?") || Strings.isNullOrBlank(columnValue)) {
         return null;
      }
      Double value = Math2.tryParseDouble(columnValue);
      if (value != null) {
         return Feature.realFeature(featureName, value);
      }
      return Feature.booleanFeature(featureName, columnValue);
   }

   @Override
   public MStream<Example> read(Resource location) throws IOException {
      List<Example> examples = new ArrayList<>();
      try (CSVReader reader = csv.reader(location)) {
         final List<String> headers = reader.getHeader().isEmpty()
                                      ? new ArrayList<>()
                                      : reader.getHeader();
         final int li = labelColumnIndex >= 0
                        ? labelColumnIndex
                        : headers.indexOf(labelColumn);

         reader.forEach(row -> {
            if (row.size() > 0 && row.stream().anyMatch(Strings::isNotNullOrBlank)) {
               List<Feature> features = new ArrayList<>();
               String label = li >= 0 ? row.get(li) : null;
               for (int i = 0; i < row.size(); i++) {
                  if (i != li) {
                     while (headers.size() <= i) {
                        headers.add("AutoColumn_" + i);
                     }
                     Feature feature = createFeature(headers.get(i), row.get(i));
                     if (feature != null) {
                        features.add(feature);
                     }
                  }
               }
               examples.add(new Instance(label, features));
            }
         });
      }
      return StreamingContext.local().stream(examples);
   }


   /**
    * Sets the column index representing the label.
    *
    * @param labelIndex the label index
    */
   public void setLabel(int labelIndex) {
      this.labelColumnIndex = labelIndex;
      this.labelColumn = null;
   }


   /**
    * Sets the name of the column representing label.
    *
    * @param columnName the column name
    */
   public void setLabel(String columnName) {
      this.labelColumnIndex = -1;
      this.labelColumn = columnName;
   }


}//END OF CSVDataSource
