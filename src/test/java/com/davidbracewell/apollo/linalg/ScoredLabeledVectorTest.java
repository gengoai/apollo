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

import com.davidbracewell.conversion.Cast;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * @author David B. Bracewell
 */
public class ScoredLabeledVectorTest extends AbstractVectorTest {

   @Before
   public void setUp() throws Exception {
      v = new ScoredLabelVector("LABELED", SparseVector.ones(10), 20);
   }


   @Test
   public void score() throws Exception {
      ScoredLabelVector sv = Cast.as(v);
      assertEquals(20, sv.getScore(), 0);
      sv.setLabel("ABC");
      assertEquals("ABC", sv.getLabel());
   }

   @Override
   public void getLabel() throws Exception {
      assertEquals("LABELED", v.getLabel());
   }

   @Test
   public void isDense() throws Exception {
      assertFalse(v.isDense());
   }

   @Test
   public void isSparse() throws Exception {
      assertTrue(v.isSparse());
   }

}