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
import com.davidbracewell.apollo.ml.preprocess.Preprocessor;
import com.davidbracewell.apollo.ml.preprocess.PreprocessorList;
import com.davidbracewell.collection.counter.Counters;
import com.davidbracewell.io.Resources;
import com.davidbracewell.io.resource.Resource;
import com.davidbracewell.io.structured.json.JSONReader;
import com.davidbracewell.io.structured.json.JSONWriter;
import org.junit.Test;

import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;

import static org.junit.Assert.*;

/**
 * @author David B. Bracewell
 */
public class TFIDFTransformTest extends BaseInstancePreprocessorTest {

   @Override
   public Dataset<Instance> getData(Preprocessor<Instance> preprocessor) {
      return Dataset.classification()
                    .source(Arrays.asList(Instance.create(Counters.newCounter("the", "black", "train"), "A"),
                                          Instance.create(Counters.newCounter("the", "black", "car"), "A"),
                                          Instance.create(Counters.newCounter("the", "blue", "bus"), "A"),
                                          Instance.create(Counters.newCounter("the", "blue", "boat"), "A")
                                         )).preprocess(PreprocessorList.create(preprocessor));
   }

   @Test
   public void test() throws Exception {
      TFIDFTransform transform = new TFIDFTransform();

      assertFalse(transform.trainOnly());

      Dataset<Instance> ds = getData(transform).encode();
      Collection<Object> featureNames = ds.getFeatureEncoder().values();
      assertFalse(featureNames.contains("the"));
      assertTrue(featureNames.contains("black"));
      assertTrue(featureNames.contains("blue"));
      assertTrue(featureNames.contains("train"));
      assertTrue(featureNames.contains("car"));
      assertTrue(featureNames.contains("bus"));
      assertTrue(featureNames.contains("boat"));

      Instance ii = ds.stream().first().orElse(Instance.create(Collections.emptyList()));
      assertEquals(0.23, ii.getValue("black"), 0.01);
      assertEquals(0, ii.getValue("the"), 0.01);
      assertEquals(0.46, ii.getValue("train"), 0.01);
   }

   @Test
   public void readWrite() throws Exception {
      TFIDFTransform transform = new TFIDFTransform();
      Resource out = Resources.fromString();
      try (JSONWriter writer = new JSONWriter(out)) {
         writer.beginDocument();
         writer.beginObject("filter");
         transform.write(writer);
         writer.endObject();
         writer.endDocument();
      }
      transform = new TFIDFTransform();
      try (JSONReader reader = new JSONReader(out)) {
         reader.beginDocument();
         reader.beginObject("filter");
         transform.read(reader);
         reader.endObject();
         reader.endDocument();
      }


      Dataset<Instance> ds = getData(transform).encode();
      Collection<Object> featureNames = ds.getFeatureEncoder().values();
      assertFalse(featureNames.contains("the"));
      assertTrue(featureNames.contains("black"));
      assertTrue(featureNames.contains("blue"));
      assertTrue(featureNames.contains("train"));
      assertTrue(featureNames.contains("car"));
      assertTrue(featureNames.contains("bus"));
      assertTrue(featureNames.contains("boat"));

      Instance ii = ds.stream().first().orElse(Instance.create(Collections.emptyList()));
      assertEquals(0.23, ii.getValue("black"), 0.01);
      assertEquals(0, ii.getValue("the"), 0.01);
      assertEquals(0.46, ii.getValue("train"), 0.01);
   }
}