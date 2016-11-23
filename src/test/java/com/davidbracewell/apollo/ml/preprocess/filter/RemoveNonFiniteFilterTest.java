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
 */

package com.davidbracewell.apollo.ml.preprocess.filter;

import com.davidbracewell.apollo.ml.Feature;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.data.Dataset;
import com.davidbracewell.apollo.ml.data.InMemoryDataset;
import com.davidbracewell.apollo.ml.data.source.DenseCSVDataSource;
import com.davidbracewell.apollo.ml.preprocess.BaseInstancePreprocessorTest;
import com.davidbracewell.apollo.ml.preprocess.Preprocessor;
import com.davidbracewell.apollo.ml.preprocess.PreprocessorList;
import com.davidbracewell.conversion.Cast;
import com.davidbracewell.io.Resources;
import com.davidbracewell.io.resource.Resource;
import com.davidbracewell.io.structured.json.JSONReader;
import com.davidbracewell.io.structured.json.JSONWriter;
import lombok.SneakyThrows;
import org.junit.Test;

import java.util.Collection;
import java.util.Collections;

import static org.junit.Assert.*;

/**
 * @author David B. Bracewell
 */
public class RemoveNonFiniteFilterTest extends BaseInstancePreprocessorTest {

   @SneakyThrows
   public Dataset<Instance> getData(Preprocessor<Instance> preprocessor) {
      DenseCSVDataSource irisData = new DenseCSVDataSource(Resources.fromClasspath(
         "com/davidbracewell/apollo/ml/iris.csv"), true);
      irisData.setLabelName("class");
      InMemoryDataset<Instance> ds = Cast.as(Dataset.classification()
                    .source(irisData));
      ds.add(Instance.create(Collections.singletonList(Feature.real("A", Double.NaN)), "iris-versacolor"));
      return ds.preprocess(PreprocessorList.create(preprocessor));
   }

   @Test
   public void test() throws Exception {
      RemoveNonFinite filter = new RemoveNonFinite();
      assertTrue(filter.trainOnly());

      InMemoryDataset<Instance> ds = Cast.as(getData(filter));

      ds.encode();
      Collection<Object> featureNames = ds.getFeatureEncoder().values();
      assertTrue(featureNames.contains("petal_length"));
      assertTrue(featureNames.contains("petal_width"));
      assertTrue(featureNames.contains("sepal_length"));
      assertTrue(featureNames.contains("sepal_width"));
      assertFalse(featureNames.contains("A"));
   }

   @Test
   public void readWrite() throws Exception {
      RemoveNonFinite filter = new RemoveNonFinite();
      Resource out = Resources.fromString();
      try (JSONWriter writer = new JSONWriter(out)) {
         writer.beginDocument();
         writer.beginObject("filter");
         filter.write(writer);
         writer.endObject();
         writer.endDocument();
      }
      filter = new RemoveNonFinite();
      try (JSONReader reader = new JSONReader(out)) {
         reader.beginDocument();
         reader.beginObject("filter");
         filter.read(reader);
         reader.endObject();
         reader.endDocument();
      }
      Dataset<Instance> ds = getData(filter);
      ds.encode();
      Collection<Object> featureNames = ds.getFeatureEncoder().values();
      assertTrue(featureNames.contains("petal_length"));
      assertTrue(featureNames.contains("petal_width"));
      assertTrue(featureNames.contains("sepal_length"));
      assertTrue(featureNames.contains("sepal_width"));
   }
}