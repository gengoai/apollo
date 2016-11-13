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

package com.davidbracewell.apollo.ml.clustering.topic;

import com.davidbracewell.apollo.ml.Feature;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.Learner;
import com.davidbracewell.apollo.ml.clustering.Cluster;
import com.davidbracewell.apollo.ml.data.Dataset;
import com.davidbracewell.collection.Streams;
import com.davidbracewell.collection.counter.Counter;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import static org.junit.Assert.*;

/**
 * @author David B. Bracewell
 */
public class GibbsLDATest {

   @Test
   public void test() throws Exception {
      GibbsLDA clusterer = Learner.<LDAModel>clustering()
                              .learnerClass(GibbsLDA.class)
                              .parameter("k", 2)
                              .parameter("maxIterations", 100)
                              .parameter("burnIn", 0)
                              .parameter("sampleLag", 0)
                              .parameter("keepDocumentTopicAssignments", true)
                              .build();


      String[] documents = {
         "The gold truck followed the brown car down the road",
         "The silver car stopped at the park",
         "The blue sky was bright",
         "I am John",
         "I am a student of St Andrew’s High School",
         "I am working in Microsoft Corporation",
         "I am doing business",
         "I am looking for a job",
         "I am a housewife",
         "He is my father",
         "She is my mother",
         "He is my elder brother",
         "This is my younger brother",
         "She is my elder sister",
         "She is my younger sister",
         "He is my grandfather",
         "She is my grandmother",
         "He is my neighbour",
         "He is my classmate",
         "He is my colleague",
         "He is my classmate and my brother",
         "The gold truck followed the brown car down the road",
         "The silver car stopped at the park",
         "The blue sky was bright",
         "I am John",
         "I am a student of St Andrew’s High School",
         "I am working in Microsoft Corporation",
         "I am doing business",
         "I am looking for a job",
         "I am a housewife",
         "He is my father",
         "She is my mother",
         "He is my elder brother",
         "This is my younger brother",
         "She is my elder sister",
         "She is my younger sister",
         "He is my grandfather",
         "She is my grandmother",
         "He is my neighbour",
         "He is my classmate",
         "He is my colleague",
         "He is my classmate and my brother"
      };
      List<Instance> instances = new ArrayList<>();
      for (String document : documents) {
         instances.add(Instance.create(Streams.asStream(Arrays.asList(document.toLowerCase().split("\\s+")))
                                              .map(Feature::TRUE)
                                              .collect(Collectors.toSet())
                                      ));

      }
      Dataset<Instance> dataset = Dataset.classification().localSource(instances.stream()).build();

      LDAModel model = clusterer.train(dataset);

      assertEquals(2, model.size());

      Cluster c1 = model.get(0);
      assertEquals(0, c1.getScore(), 1);
      assertEquals(42, c1.size(), 10);
      Cluster c2 = model.get(1);
      assertEquals(0, c2.getScore(), 1);
      assertEquals(42, c2.size(), 10);

      Counter<String> c1Words = model.getTopicWords(0);
      assertEquals(model.numberOfFeatures(), c1Words.size(), 10);
      Counter<String> c2Words = model.getTopicWords(0);
      assertEquals(model.numberOfFeatures(), c2Words.size(), 10);

      Instance ii = Instance.create(Streams.asStream(
         Arrays.asList("My name is Joe and I am my own grandfather".split("\\s+")))
                                           .map(Feature::TRUE)
                                           .collect(Collectors.toSet())
                                   );

      double[] dist = model.softCluster(ii);
      assertEquals(2, dist.length);
      assertEquals(0.54, dist[0], 0.1);
      assertEquals(0.52, dist[1], 0.1);
   }
}