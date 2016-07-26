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

package com.davidbracewell.apollo.ml.clustering.flat;

import com.davidbracewell.apollo.affinity.Distance;
import com.davidbracewell.apollo.ml.clustering.ClustererTest;
import com.davidbracewell.apollo.ml.clustering.Clustering;
import org.junit.Test;

/**
 * @author David B. Bracewell
 */
public class FuzzyKMeansTest extends ClustererTest {

  public FuzzyKMeansTest() {
    super(new FuzzyKMeans(Distance.Manhattan, 2, 100, 1.1, 100));
  }

  @Test
  public void testCluster() throws Exception {
    Clustering c = cluster();
    System.err.println(c.get(0));
//    System.err.println(c.get(1));
  }
}