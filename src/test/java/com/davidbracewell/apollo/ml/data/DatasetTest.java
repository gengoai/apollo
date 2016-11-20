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

package com.davidbracewell.apollo.ml.data;

import com.davidbracewell.apollo.ml.Feature;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.preprocess.PreprocessorList;
import com.davidbracewell.apollo.ml.preprocess.filter.NameFilter;
import com.davidbracewell.io.resource.Resource;
import com.davidbracewell.io.resource.StringResource;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static org.junit.Assert.*;

/**
 * @author David B. Bracewell
 */
public abstract class DatasetTest {

   Dataset<Instance> dataset;


   protected List<Instance> createInstances() {
      List<Instance> instances = new ArrayList<>();
      for (int i = 0; i < 10; i++) {
         instances.add(Instance.create(Collections.singletonList(Feature.TRUE("true")), "true"));
      }
      for (int i = 0; i < 20; i++) {
         instances.add(Instance.create(Arrays.asList(Feature.TRUE("false"), Feature.TRUE("toDelete")), "false"));
      }
      return instances;
   }

   @Test
   public void test() throws Exception {
      long trueCount = dataset.stream().filter(ii -> ii.getLabel().equals("true")).count();
      long falseCount = dataset.stream().filter(ii -> ii.getLabel().equals("false")).count();
      assertEquals(10, trueCount);
      assertEquals(20, falseCount);
   }

   @Test
   public void undersample() throws Exception {
      Dataset<Instance> under = dataset.undersample();
      long trueCount = under.stream().filter(ii -> ii.getLabel().equals("true")).count();
      long falseCount = under.stream().filter(ii -> ii.getLabel().equals("false")).count();
      assertEquals(10, trueCount);
      assertEquals(10, falseCount);
   }

   @Test
   public void oversample() throws Exception {
      Dataset<Instance> under = dataset.oversample();
      long trueCount = under.stream().filter(ii -> ii.getLabel().equals("true")).count();
      long falseCount = under.stream().filter(ii -> ii.getLabel().equals("false")).count();
      assertEquals(20, trueCount);
      assertEquals(20, falseCount);
   }


   @Test
   public void iterator() throws Exception {
      int count = 0;
      for (Instance aDataset : dataset) {
         count++;
      }
      assertEquals(30, count);
   }

   @Test
   public void readWrite() throws Exception {
      Resource resource = new StringResource();
      Dataset<Instance> copy = dataset.copy();
      copy = copy.preprocess(PreprocessorList.create(new NameFilter("toDelete")));
      copy.write(resource);
      copy = Dataset.classification().load(resource).build();
      long trueCount = copy.stream().filter(ii -> ii.getLabel().equals("true")).count();
      long falseCount = copy.stream().filter(ii -> ii.getLabel().equals("false")).count();
      long toDeleteCount = copy.stream().filter(ii -> ii.getFeatureSpace().anyMatch(s -> s.equals("toDelete"))).count();
      assertEquals(10, trueCount);
      assertEquals(20, falseCount);
      assertEquals(0, toDeleteCount);
   }
}