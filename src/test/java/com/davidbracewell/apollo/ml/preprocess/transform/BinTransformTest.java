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

package com.davidbracewell.apollo.ml.preprocess.transform;

import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.data.Dataset;
import com.davidbracewell.apollo.ml.preprocess.BaseInstancePreprocessorTest;
import com.davidbracewell.io.Resources;
import com.davidbracewell.io.resource.Resource;
import com.davidbracewell.json.JsonReader;
import com.davidbracewell.json.JsonWriter;
import org.junit.Test;

import java.util.Collection;
import java.util.Collections;

import static org.junit.Assert.*;

/**
 * @author David B. Bracewell
 */
public class BinTransformTest extends BaseInstancePreprocessorTest {
   @Test
   public void test() throws Exception {
      BinTransform transform = new BinTransform("sepal_length", 5);

      assertFalse(transform.trainOnly());

      Dataset<Instance> ds = getData(transform).encode();
      Collection<Object> featureNames = ds.getFeatureEncoder().values();
      assertTrue(featureNames.contains("petal_length"));
      assertTrue(featureNames.contains("petal_width"));
      assertTrue(featureNames.contains("sepal_length=0"));
      assertTrue(featureNames.contains("sepal_length=1"));
      assertTrue(featureNames.contains("sepal_length=2"));
      assertTrue(featureNames.contains("sepal_length=3"));
      assertTrue(featureNames.contains("sepal_length=4"));
      assertTrue(featureNames.contains("sepal_width"));

      Instance ii = ds.stream().first().orElse(Instance.create(Collections.emptyList()));
      assertEquals(1, ii.getValue("sepal_length=1"), 0.01);

      transform = new BinTransform(3);
      ds = getData(transform);
      ds.encode();
      featureNames = ds.getFeatureEncoder().values();
      assertTrue(featureNames.contains("petal_length=0"));
      assertTrue(featureNames.contains("petal_length=1"));
      assertTrue(featureNames.contains("petal_length=2"));
      assertTrue(featureNames.contains("petal_width=0"));
      assertTrue(featureNames.contains("sepal_length=1"));
      assertTrue(featureNames.contains("sepal_length=2"));
      assertTrue(featureNames.contains("sepal_width=0"));
      assertTrue(featureNames.contains("sepal_width=1"));

      ii = ds.stream().first().orElse(Instance.create(Collections.emptyList()));
      assertEquals(1, ii.getValue("sepal_length=1"), 0.01);
      assertEquals(1, ii.getValue("sepal_width=1"), 0.01);
      assertEquals(1, ii.getValue("petal_length=0"), 0.01);
      assertEquals(1, ii.getValue("petal_width=0"), 0.01);
   }

   @Test
   public void readWrite() throws Exception {
      BinTransform transform = new BinTransform("sepal_length", 5);
      Resource out = Resources.fromString();
      try (JsonWriter writer = new JsonWriter(out)) {
         writer.beginDocument();
         writer.beginObject("filter");
         transform.toJson(writer);
         writer.endObject();
         writer.endDocument();
      }
      transform = new BinTransform(1);
      try (JsonReader reader = new JsonReader(out)) {
         reader.beginDocument();
         reader.beginObject("filter");
         transform.fromJson(reader);
         reader.endObject();
         reader.endDocument();
      }
      Dataset<Instance> ds = getData(transform).encode();
      Collection<Object> featureNames = ds.getFeatureEncoder().values();
      assertTrue(featureNames.contains("petal_length"));
      assertTrue(featureNames.contains("petal_width"));
      assertTrue(featureNames.contains("sepal_length=0"));
      assertTrue(featureNames.contains("sepal_length=1"));
      assertTrue(featureNames.contains("sepal_length=2"));
      assertTrue(featureNames.contains("sepal_length=3"));
      assertTrue(featureNames.contains("sepal_length=4"));
      assertTrue(featureNames.contains("sepal_width"));

      Instance ii = ds.stream().first().orElse(Instance.create(Collections.emptyList()));
      assertEquals(1, ii.getValue("sepal_length=1"), 0.01);
   }
}