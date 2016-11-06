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

package com.davidbracewell.apollo.distribution;

import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * @author David B. Bracewell
 */
public class UniformDiscreteTest {

   UniformDiscrete uniformDiscrete;

   @Before
   public void setUp() throws Exception {
      uniformDiscrete = new UniformDiscrete(1, 5);
   }

   @Test
   public void probability() throws Exception {
      assertEquals(0, uniformDiscrete.probability(0), 0.01);
      assertEquals(0, uniformDiscrete.probability(6), 0.01);
      assertEquals(0.2, uniformDiscrete.probability(1), 0.01);
   }

   @Test
   public void setters() throws Exception {
      UniformDiscrete ud = new UniformDiscrete(5);
      ud.setMax(10);
      ud.setMin(0);
      assertEquals(0.2, uniformDiscrete.probability(1), 0.01);
   }

   @Test
   public void mode() throws Exception {
      assertTrue(Double.isNaN(uniformDiscrete.getMode()));
   }

   @Test
   public void variance() throws Exception {
      assertEquals(2, uniformDiscrete.getVariance(), 0.1);

   }

   @Test
   public void mean() throws Exception {
      assertEquals(3, uniformDiscrete.getMean(), 0);
   }

   @Test
   public void max() throws Exception {
      assertEquals(10, new UniformDiscrete(10).getMax());
   }

   @Test
   public void inverseCumulativeProbability() throws Exception {
      assertEquals(1, uniformDiscrete.inverseCumulativeProbability(0.2), 0.01);
   }

   @Test
   public void cumulativeProbability() throws Exception {
      assertEquals(0.2, uniformDiscrete.cumulativeProbability(4, 5), 0.01);
   }
}