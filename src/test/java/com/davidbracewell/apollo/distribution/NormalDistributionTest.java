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
public class NormalDistributionTest {

   NormalDistribution normal;

   @Before
   public void setUp() throws Exception {
      normal = new NormalDistribution();
      normal.addValue(1);
      normal.addValue(2);
      normal.addValue(1);
      normal.addValue(0);
   }

   @Test
   public void probability() throws Exception {
      assertEquals(0.49, normal.probability(1), 0.01);
      assertEquals(0.5, normal.cumulativeProbability(1), 0.01);
      assertEquals(0.78, normal.cumulativeProbability(0, 2), 0.01);
      assertEquals(1, normal.inverseCumulativeProbability(0.5), 0.01);
   }

   @Test
   public void stats() throws Exception {
      assertEquals(1, normal.getMode(), 0);
      assertEquals(1, normal.getMean(), 0);
      assertEquals(0.67, normal.getVariance(), 0.1);
   }
}