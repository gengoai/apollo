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

package com.davidbracewell.apollo.linalg.decompose;

import com.davidbracewell.apollo.linalg.DenseMatrix;
import com.davidbracewell.apollo.linalg.Matrix;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * @author David B. Bracewell
 */
public class SingularValueDecompositionTest {

   Matrix m;

   @Before
   public void setUp() throws Exception {
      m = new DenseMatrix(new double[][]{
         new double[]{1, 0, 1, 0, 0},
         new double[]{1, 1, 0, 0, 0},
         new double[]{0, 1, 0, 0, 0},
         new double[]{0, 1, 1, 0, 0},
         new double[]{0, 0, 0, 1, 0},
         new double[]{0, 0, 1, 1, 0},
         new double[]{0, 0, 0, 1, 0},
         new double[]{0, 0, 0, 1, 1},
      });
   }


   @Test
   public void decompose() throws Exception {
      SVD svd = new SVD(false);
      Matrix[] uSv = svd.decompose(m);
      assertEquals(2.29, uSv[1].get(0, 0), 0.01);
      assertEquals(2.01, uSv[1].get(1, 1), 0.01);
      assertEquals(1.36, uSv[1].get(2, 2), 0.01);
      assertEquals(1.12, uSv[1].get(3, 3), 0.01);
      assertEquals(0.80, uSv[1].get(4, 4), 0.01);
   }
}