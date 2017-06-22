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

package com.davidbracewell.apollo.ml.embedding;

import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.linalg.VectorCompositions;
import com.davidbracewell.apollo.ml.data.Dataset;
import com.davidbracewell.apollo.ml.data.DatasetType;
import com.davidbracewell.apollo.ml.sequence.Sequence;
import com.davidbracewell.config.Config;
import com.davidbracewell.stream.StreamingContext;
import org.junit.Test;

import java.util.List;
import java.util.stream.Stream;

import static com.davidbracewell.tuple.Tuples.$;
import static org.junit.Assert.*;

/**
 * @author David B. Bracewell
 */
public class SparkWord2VecTest {

   @Test
   public void build() throws Exception {
      Config.initializeTest();
      Config.setProperty("spark.master", "local[*]");
      SparkWord2Vec word2Vec = new SparkWord2Vec();
      word2Vec.setMinCount(1);
      word2Vec.setDimension(5);
      word2Vec.setRandomSeed(1);
      Dataset<Sequence> dataset = Dataset.embedding(DatasetType.Distributed,
                                                    StreamingContext.distributed().stream("The blue train",
                                                                                          "The black train",
                                                                                          "The red boat",
                                                                                          "The green boat"),
                                                    sentence -> Stream.of(sentence.toLowerCase().split("\\s+"))
                                                   );

      Embedding model = word2Vec.train(dataset);

      assertTrue(model.contains("the"));
      assertTrue(model.contains("black"));


      List<Vector> n = model.nearest("black", 2);
      assertEquals(2, n.size());
      assertTrue(n.get(0).getWeight() >= 0.1);
      assertTrue(n.get(1).getWeight() >= 0.1);


      Vector blackAndRed = model.compose(VectorCompositions.Sum, "black", "red");
      assertArrayEquals(model.getVector("black").add(model.getVector("red")).toArray(), blackAndRed.toArray(), 0.001);


      n = model.nearest($("black"), 2);
      assertEquals(2, n.size());
      assertTrue(n.get(0).getWeight() >= 0.1);
      assertTrue(n.get(1).getWeight() >= 0.1);


   }
}