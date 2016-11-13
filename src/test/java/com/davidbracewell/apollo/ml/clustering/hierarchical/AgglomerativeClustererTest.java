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

package com.davidbracewell.apollo.ml.clustering.hierarchical;

import com.davidbracewell.apollo.ml.clustering.Cluster;
import com.davidbracewell.apollo.ml.clustering.ClustererTest;
import com.davidbracewell.apollo.ml.clustering.Clustering;
import com.davidbracewell.conversion.Cast;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * @author David B. Bracewell
 */
public class AgglomerativeClustererTest extends ClustererTest {

   public AgglomerativeClustererTest() {
      super(new AgglomerativeClusterer());
   }


   @Test
   public void testCluster() throws Exception {
      HierarchicalClustering c = Cast.as(cluster());
      Cluster root = c.getRoot();
      assertEquals(2.23, root.getScore(), 0.1);
      Clustering fc = c.asFlat(1.0);
      assertEquals(2, fc.size());
   }

}