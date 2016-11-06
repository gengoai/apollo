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

package com.davidbracewell.apollo.linalg.store;

import com.davidbracewell.apollo.linalg.DenseVector;
import com.davidbracewell.apollo.linalg.LabeledVector;
import com.davidbracewell.apollo.linalg.ScoredLabelVector;
import com.davidbracewell.collection.Sets;
import org.junit.After;
import org.junit.Test;

import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.*;

/**
 * @author David B. Bracewell
 */
public abstract class AbstractVectorStoreTest {

   VectorStore<String> vs;


   public List<LabeledVector> getVectors() {
      return Arrays.asList(
         LabeledVector.sparse(10, "A").mapAddSelf(1),
         LabeledVector.sparse(10, "B").mapAddSelf(1.5),
         LabeledVector.sparse(10, "C").mapAddSelf(4.5),
         LabeledVector.sparse(10, "D").mapAddSelf(5),
         LabeledVector.sparse(10, "E").mapAddSelf(100)
                          );
   }

   @After
   public void tearDown() throws Exception {
      assertTrue(vs.remove(LabeledVector.sparse(10, "A").mapAddSelf(1)));
   }

   @Test
   public void size() throws Exception {
      assertEquals(5, vs.size());
   }


   @Test
   public void dimension() throws Exception {
      assertEquals(10, vs.dimension());
   }

   @Test
   public void nearestVector() throws Exception {
      List<ScoredLabelVector> list = vs.nearest(DenseVector.ones(10));
      assertEquals(2.2E-16, list.get(0).getScore(), 0.1);
   }

   @Test(expected = IllegalArgumentException.class)
   public void nearestVectorError() throws Exception {
      List<ScoredLabelVector> list = vs.nearest(DenseVector.ones(100));
   }

   @Test
   public void nearestVectorK() throws Exception {
      List<ScoredLabelVector> list = vs.nearest(DenseVector.ones(10), 2);
      assertEquals(2, list.size());
      assertEquals(0, list.get(0).getScore(), 0);
      assertEquals(0, list.get(1).getScore(), 0.01);
   }

   @Test(expected = IllegalArgumentException.class)
   public void nearestVectorKError() throws Exception {
      List<ScoredLabelVector> list = vs.nearest(DenseVector.ones(100), 1);
   }

   @Test
   public void nearestVectorKThreshold() throws Exception {
      List<ScoredLabelVector> list = vs.nearest(DenseVector.ones(10), 2, 2);
      assertEquals(2, list.size());
      assertEquals(0, list.get(0).getScore(), 0);
      assertEquals(0, list.get(1).getScore(), 0.01);
   }

   @Test(expected = IllegalArgumentException.class)
   public void nearestVectorKThresholdError() throws Exception {
      List<ScoredLabelVector> list = vs.nearest(DenseVector.ones(100), 1, 4);
   }

   @Test
   public void keySet() throws Exception {
      assertEquals(Sets.set("A", "B", "C", "D", "E"), vs.keySet());
   }

   @Test
   public void get() throws Exception {
      assertEquals(LabeledVector.sparse(10, "A").mapAddSelf(1),
                   vs.get("A")
                  );

   }

   @Test
   public void containsKey() throws Exception {
      assertTrue(vs.containsKey("A"));
      assertFalse(vs.containsKey("Z"));
   }

}