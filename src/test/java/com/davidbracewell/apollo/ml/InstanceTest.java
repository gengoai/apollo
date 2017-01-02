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

package com.davidbracewell.apollo.ml;

import com.davidbracewell.Interner;
import com.davidbracewell.collection.Sets;
import org.junit.Before;
import org.junit.Test;

import java.util.Arrays;
import java.util.Collections;
import java.util.stream.Collectors;

import static org.junit.Assert.*;

/**
 * @author David B. Bracewell
 */
public class InstanceTest {

   Instance ii;

   @Before
   public void setUp() throws Exception {
      ii = Instance.create(Arrays.asList(Feature.TRUE("F1"),
                                         Feature.real("F2", 2.0),
                                         Feature.TRUE("PREFIX", "WORD")
                                        ),
                           "LABEL");

   }

   @Test
   public void asInstances() throws Exception {
      assertEquals(Collections.singletonList(ii), ii.asInstances());
   }

   @Test
   public void copy() throws Exception {
      assertEquals(ii, ii.copy());
   }

   @Test
   public void getFeatureSpace() throws Exception {
      assertEquals(Sets.set("F1", "F2", "PREFIX=WORD"), ii.getFeatureSpace().collect(Collectors.toSet()));
   }

   @Test
   public void getFeatures() throws Exception {
      assertEquals(Arrays.asList(Feature.TRUE("F1"),
                                 Feature.real("F2", 2.0),
                                 Feature.TRUE("PREFIX", "WORD")
                                ),
                   ii.getFeatures()
                  );
   }

   @Test
   public void getLabel() throws Exception {
      assertEquals("LABEL", ii.getLabel());
   }

   @Test
   public void setLabel() throws Exception {
      Instance ii2 = Instance.create(Arrays.asList(Feature.TRUE("F1"),
                                                   Feature.real("F2", 2.0),
                                                   Feature.TRUE("PREFIX", "WORD")
                                                  ),
                                     "LABEL");
      ii2.setLabel("NEW LABEL");
      assertEquals("LABEL", ii.getLabel());
   }

   @Test
   public void getLabelSet() throws Exception {
      assertEquals(Collections.singleton("LABEL"), ii.getLabelSet());
   }

   @Test
   public void getLabelSpace() throws Exception {
      assertEquals(Collections.singleton("LABEL"), ii.getLabelSpace().collect(Collectors.toSet()));
   }

   @Test
   public void getValue() throws Exception {
      assertEquals(1.0, ii.getValue("F1"), 0.0);
   }

   @Test
   public void hasLabel() throws Exception {
      assertTrue(ii.hasLabel());
   }

   @Test
   public void hasLabel1() throws Exception {
      assertTrue(ii.hasLabel("LABEL"));
   }

   @Test
   public void intern() throws Exception {
      Interner<String> interner = new Interner<>();
      assertEquals(ii, ii.intern(interner));
      assertEquals(3, interner.size());

   }

   @Test
   public void isMultiLabeled() throws Exception {
      assertFalse(ii.isMultiLabeled());
      Instance ii2 = Instance.create(Arrays.asList(Feature.TRUE("F1"),
                                                   Feature.real("F2", 2.0),
                                                   Feature.TRUE("PREFIX", "WORD")
                                                  ),
                                     Arrays.asList("LABEL", "NON-LABEL"));
      assertTrue(ii2.isMultiLabeled());
   }

   @Test
   public void iterator() throws Exception {
      int c = 0;
      for (Feature feature : ii) {
         c++;
      }
      assertEquals(3, c);
   }

   @Test
   public void read() throws Exception {
      Instance ii2 = Example.fromJson("{\"label\":\"LABEL\",\"weight\":1.0,\"features\":{\"F1\":1.0,\"F2\":2.0,\"PREFIX=WORD\":1.0}}",
                                      Instance.class);
      assertEquals(ii.toString(), ii2.toString());
   }

   @Test
   public void write() throws Exception {
      assertEquals("{\"label\":\"LABEL\",\"weight\":1.0,\"features\":{\"F1\":1.0,\"F2\":2.0,\"PREFIX=WORD\":1.0}}",
                   ii.toJson());
   }

   @Test
   public void weight() throws Exception {
      assertEquals(1.0, ii.getWeight(), 0.0);
      ii.setWeight(2.0);
      assertEquals(2.0, ii.getWeight(), 0.0);
      ii.setWeight(1.0);
      assertEquals(1.0, ii.getWeight(), 0.0);
   }

}