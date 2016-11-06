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

package com.davidbracewell.apollo.ml;

import com.davidbracewell.apollo.linalg.DenseVector;
import com.davidbracewell.apollo.linalg.LabeledVector;
import com.davidbracewell.apollo.ml.data.Dataset;
import lombok.SneakyThrows;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static junit.framework.TestCase.*;

/**
 * @author David B. Bracewell
 */
public abstract class AbstractEncoderTest {

   Encoder encoder;
   Dataset<Instance> dataset = getData();

   @SneakyThrows
   public Dataset<Instance> getData() {
      List<Instance> data = new ArrayList<>();
      for (int i = 0; i < 1_000; i++) {
         data.add(Instance.fromVector(new LabeledVector("true", DenseVector.ones(20))));
      }
      for (int i = 0; i < 1_000; i++) {
         data.add(Instance.fromVector(new LabeledVector("false", DenseVector.zeros(20))));
      }
      return Dataset.classification()
                    .data(data)
                    .featureEncoder(new IndexEncoder())
                    .build()
                    .shuffle(new Random(1234));
   }



   @Test
   public void get() throws Exception {
      encoder.fit(dataset);
      assertTrue(encoder.get("1") != -1);
   }

   @Test
   public void index() throws Exception {
      encoder.fit(dataset);
      assertTrue(encoder.index("1") != -1);
   }

   @Test
   public void encode() throws Exception {
      encoder.fit(dataset);
      assertTrue(encoder.encode("1") != -1);
   }

   @Test
   public void decode() throws Exception {
      encoder.fit(dataset);
      int id = encoder.index("1");
      assertEquals("1", encoder.decode(id));
   }

   @Test
   public void freeze() throws Exception {
      encoder.fit(dataset);
      Encoder e = encoder.createNew();
      e.freeze();
      assertTrue(e.isFrozen());
   }

   @Test
   public void unFreeze() throws Exception {
      encoder.fit(dataset);
      Encoder e = encoder.createNew();
      e.freeze();
      e.unFreeze();
      assertFalse(e.isFrozen());
   }


}//END OF AbstractEncoderTest
