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
import com.davidbracewell.apollo.ml.data.Dataset;
import com.davidbracewell.stream.StreamingContext;
import lombok.SneakyThrows;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static org.junit.Assert.*;

/**
 * @author David B. Bracewell
 */
public class RegressionLabelEncoderTest {

   Encoder encoder = new RegressionLabelEncoder();

   @SneakyThrows
   public Dataset<Instance> getData() {
      List<Instance> data = new ArrayList<>();
      for (int i = 0; i < 1_000; i++) {
         data.add(Instance.fromVector(DenseVector.ones(20).setLabel("true")));
      }
      for (int i = 0; i < 1_000; i++) {
         data.add(Instance.fromVector(DenseVector.zeros(20).setLabel("false")));
      }
      return Dataset.classification()
                    .featureEncoder(new IndexEncoder())
                    .source(data)
                    .shuffle(new Random(1234));
   }

   @Test
   public void createNew() throws Exception {
      assertTrue(encoder.createNew() instanceof RegressionLabelEncoder);
   }

   @Test
   public void decode() throws Exception {
      assertEquals(20d, encoder.decode(20));
   }

   @Test(expected = IllegalArgumentException.class)
   public void encodeError() throws Exception {
      assertEquals(-1, encoder.encode("20"), 0);
   }

   @Test
   public void encode() throws Exception {
      assertEquals(20.0, encoder.encode(20), 0);
   }


   @Test
   public void fit() throws Exception {
      encoder.fit(StreamingContext.local().stream("20"));
   }

   @Test
   public void fit1() throws Exception {
      encoder.fit(getData());
   }

   @Test
   public void freeze() throws Exception {
      assertTrue(encoder.isFrozen());
      encoder.unFreeze();
      assertTrue(encoder.isFrozen());
      encoder.freeze();
      assertTrue(encoder.isFrozen());
   }

   @Test(expected = IllegalArgumentException.class)
   public void getError() throws Exception {
      assertEquals(-1, encoder.get("20"), 0);
   }

   @Test
   public void get() throws Exception {
      assertEquals(20.0, encoder.get(20), 0);
   }


   @Test
   public void size() throws Exception {
      assertEquals(0, encoder.size());
   }

   @Test
   public void values() throws Exception {
      assertTrue(encoder.values().isEmpty());
   }

}