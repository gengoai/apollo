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

package com.davidbracewell.apollo.linalg;

import org.junit.Before;
import org.junit.Test;

import java.util.Map;

import static org.junit.Assert.*;

/**
 * @author David B. Bracewell
 */
public class VectorMapTest {

   Map<Integer, Double> map;

   @Before
   public void setUp() throws Exception {
      Vector v = SparseVector.ones(100);
      v.set(54, 0);
      map = VectorMap.wrap(v);
   }

   @Test
   public void entrySet() throws Exception {
      assertEquals(99, map.entrySet().size());
   }

   @Test
   public void keySet() throws Exception {
      assertEquals(99, map.keySet().size());
      assertTrue(map.keySet().contains(67));
   }

   @Test
   public void size() throws Exception {
      assertEquals(99, map.size());
   }

   @Test
   public void containsKey() throws Exception {
      assertTrue(map.containsKey(67));
      assertFalse(map.containsKey("A"));
      assertFalse(map.containsKey(54));
   }

   @Test
   public void get() throws Exception {
      assertEquals(1, map.get(1), 0);
   }

}