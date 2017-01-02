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

import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.data.Dataset;
import com.davidbracewell.apollo.ml.preprocess.BaseInstancePreprocessorTest;
import com.davidbracewell.io.Resources;
import com.davidbracewell.io.resource.Resource;
import com.davidbracewell.io.structured.json.JSONReader;
import com.davidbracewell.io.structured.json.JSONWriter;
import org.junit.Test;

import java.util.Collection;

import static org.junit.Assert.*;

/**
 * @author David B. Bracewell
 */
public class MinCountFilterTest extends BaseInstancePreprocessorTest {

   @Test
   public void test() throws Exception {
      MinCountFilter filter = new MinCountFilter("sepal", 1);


      assertTrue(filter.trainOnly());

      Dataset<Instance> ds = getData(filter);
      ds.encode();
      Collection<Object> featureNames = ds.getFeatureEncoder().values();
      assertTrue(featureNames.contains("petal_length"));
      assertTrue(featureNames.contains("petal_width"));
      assertTrue(featureNames.contains("sepal_length"));
      assertTrue(featureNames.contains("sepal_width"));

      filter = new MinCountFilter(1);
      ds = getData(filter);
      ds.encode();
      featureNames = ds.getFeatureEncoder().values();
      assertTrue(featureNames.contains("petal_length"));
      assertTrue(featureNames.contains("petal_width"));
      assertTrue(featureNames.contains("sepal_length"));
      assertTrue(featureNames.contains("sepal_width"));
   }

   @Test
   public void readWrite() throws Exception {
      MinCountFilter filter = new MinCountFilter("sepal", 1);
      Resource out = Resources.fromString();
      try (JSONWriter writer = new JSONWriter(out)) {
         writer.beginDocument();
         writer.beginObject("filter");
         filter.write(writer);
         writer.endObject();
         writer.endDocument();
      }
      filter = new MinCountFilter();
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