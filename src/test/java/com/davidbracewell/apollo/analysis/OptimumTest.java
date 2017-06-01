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

package com.davidbracewell.apollo.analysis;

import com.davidbracewell.apollo.optimization.Optimum;
import org.junit.Test;

import static com.davidbracewell.tuple.Tuples.$;
import static org.junit.Assert.*;

/**
 * @author David B. Bracewell
 */
public class OptimumTest {
   @Test
   public void compare() throws Exception {
      assertTrue(Optimum.MAXIMUM.compare(5, 4) < 0);
      assertTrue(Optimum.MAXIMUM.compare(5, 5) == 0);
      assertTrue(Optimum.MAXIMUM.compare(4, 5) > 0);

      assertTrue(Optimum.MINIMUM.compare(5, 4) > 0);
      assertTrue(Optimum.MINIMUM.compare(5, 5) == 0);
      assertTrue(Optimum.MINIMUM.compare(4, 5) < 0);
   }

   @Test
   public void test1() throws Exception {
      assertTrue(Optimum.MAXIMUM.test(10, 9));
      assertFalse(Optimum.MAXIMUM.test(10, 25));

      assertFalse(Optimum.MINIMUM.test(10, 9));
      assertTrue(Optimum.MINIMUM.test(10, 25));
   }

   @Test
   public void startingValue() throws Exception {
      assertEquals(Double.NEGATIVE_INFINITY, Optimum.MAXIMUM.startingValue(), 0);
      assertEquals(Double.POSITIVE_INFINITY, Optimum.MINIMUM.startingValue(), 0);
   }

   @Test
   public void optimum() throws Exception {
      assertEquals($(0, 1d), Optimum.MINIMUM.optimum(new double[]{1, 2, 3}));
      assertEquals($(2, 3d), Optimum.MAXIMUM.optimum(new double[]{1, 2, 3}));
   }

   @Test
   public void optimumValue() throws Exception {
      assertEquals(1d, Optimum.MINIMUM.optimumValue(new double[]{1, 2, 3}), 0);
      assertEquals(3d, Optimum.MAXIMUM.optimumValue(new double[]{1, 2, 3}), 0);
   }

   @Test
   public void optimumIndex() throws Exception {
      assertEquals(0, Optimum.MINIMUM.optimumIndex(new double[]{1, 2, 3}), 0);
      assertEquals(2, Optimum.MAXIMUM.optimumIndex(new double[]{1, 2, 3}), 0);
   }

}