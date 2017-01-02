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

import static org.junit.Assert.*;

/**
 * @author David B. Bracewell
 */
public class VectorCompositionsTest {

   Vector v1;
   Vector v2;

   @Before
   public void setUp() throws Exception {
      v1 = SparseVector.ones(3);
      v2 = DenseVector.wrap(2, 2, 2);
   }


   @Test
   public void average() throws Exception {
      assertEquals(DenseVector.wrap(1.5, 1.5, 1.5),
                   VectorCompositions.Average.compose(3, v1, v2));

   }


   @Test
   public void sum() throws Exception {
      assertEquals(DenseVector.wrap(3, 3, 3),
                   VectorCompositions.Sum.compose(3, v1, v2));

   }


   @Test
   public void pointWiseMultiply() throws Exception {
      assertEquals(DenseVector.wrap(2, 2, 2),
                   VectorCompositions.PointWiseMultiply.compose(3, v1, v2));

   }

   @Test
   public void max() throws Exception {
      assertEquals(DenseVector.wrap(2, 2, 2),
                   VectorCompositions.Max.compose(3, v1, v2));

   }

   @Test
   public void min() throws Exception {
      assertEquals(DenseVector.wrap(1, 1, 1),
                   VectorCompositions.Min.compose(3, v1, v2));

   }


}