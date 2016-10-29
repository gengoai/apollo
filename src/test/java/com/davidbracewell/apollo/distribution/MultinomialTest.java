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

import java.util.stream.IntStream;

import static org.junit.Assert.*;

/**
 * @author David B. Bracewell
 */
public class MultinomialTest {

   Multinomial multinomial;

   @Before
   public void setUp() throws Exception {
      multinomial = new Multinomial(3, 1);
      multinomial.increment(0, 5);
      multinomial.increment(1, 2);
      multinomial.increment(2, 3);
   }

   @Test
   public void stats() throws Exception {
      assertEquals(10, multinomial.getTotalObservations(), 0);
      assertEquals(0, multinomial.getMode(), 0);
      assertEquals(0.85, multinomial.getMean(), 0.01);
      assertEquals(0.75, multinomial.getVariance(), 0.01);
   }


   @Test
   public void cumulativeProbability() throws Exception {
      assertEquals(0.54, multinomial.cumulativeProbability(0, 5), 0.01);
      assertEquals(1.0, multinomial.cumulativeProbability(5), 0.01);
   }

   @Test
   public void inverseCumulativeProbability() throws Exception {
      assertEquals(0, multinomial.inverseCumulativeProbability(0.8), 0.01);
   }


   @Test
   public void sample() throws Exception {
      int[] sample = multinomial.sample(10);
      double zeroCount = IntStream.of(sample).filter(i -> i == 0).count();
      double oneCount = IntStream.of(sample).filter(i -> i == 1).count();
      double twoCount = IntStream.of(sample).filter(i -> i == 2).count();
      assertEquals(5.0, zeroCount, 2);
      assertEquals(2.0, oneCount, 2);
      assertEquals(3.0, twoCount, 2);
   }

}