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

/**
 * @author David B. Bracewell
 */
public class TruncatedSVDTest {

//   Matrix m;
//
//   @Before
//   public void setUp() throws Exception {
//      m = new DenseMatrix(new double[][]{
//         new double[]{1, 0, 1, 0, 0},
//         new double[]{1, 1, 0, 0, 0},
//         new double[]{0, 1, 0, 0, 0},
//         new double[]{0, 1, 1, 0, 0},
//         new double[]{0, 0, 0, 1, 0},
//         new double[]{0, 0, 1, 1, 0},
//         new double[]{0, 0, 0, 1, 0},
//         new double[]{0, 0, 0, 1, 1},
//      });
//   }
//
//
//   @Test
//   public void decompose() throws Exception {
//      TruncatedSVD tSvd = new TruncatedSVD(2);
//      Matrix[] tUSv = tSvd.decompose(m);
//      assertEquals(2.29, tUSv[1].get(0, 0), 0.01);
//      assertEquals(2.01, tUSv[1].get(1, 1), 0.01);
//
//      Matrix us = tUSv[0].multiply(tUSv[1]);
//
//      //Vector of first word
//      assertEquals(-0.91, us.get(0, 0), 0.01);
//      assertEquals(-0.56, us.get(0, 1), 0.01);
//
//
//   }
}