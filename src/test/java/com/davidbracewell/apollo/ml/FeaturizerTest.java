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

import com.davidbracewell.apollo.ml.sequence.Sequence;
import com.davidbracewell.apollo.ml.sequence.SequenceFeaturizer;
import com.davidbracewell.apollo.ml.sequence.SequenceInput;
import org.junit.Before;
import org.junit.Test;

import java.util.Collections;

import static org.junit.Assert.*;

/**
 * @author David B. Bracewell
 */
public class FeaturizerTest {

   Featurizer<String> f1;


   @Before
   public void setUp() throws Exception {
      f1 = ((Featurizer<String>) s -> Collections.singleton(Feature.TRUE(s)));
   }

   @Test
   public void chain() throws Exception {
      Featurizer<String> chain = f1.and(
         ((Featurizer<String>) s -> Collections.singleton(Feature.TRUE(s.toLowerCase()))));
      Instance ii = chain.extractInstance("TeSt");
      assertEquals(2, ii.getFeatures().size());
      assertEquals(1.0, ii.getValue("TeSt"), 0.0);
      assertEquals(1.0, ii.getValue("test"), 0.0);
      assertEquals(0.0, ii.getValue("testing"), 0.0);


      chain = chain.and(((Featurizer<String>) s -> Collections.singleton(Feature.TRUE("testing"))));
      ii = chain.extractInstance("TeSt");
      assertEquals(3, ii.getFeatures().size());
      assertEquals(1.0, ii.getValue("TeSt"), 0.0);
      assertEquals(1.0, ii.getValue("test"), 0.0);
      assertEquals(1.0, ii.getValue("testing"), 0.0);

      chain = Featurizer.chain(f1);
      ii = chain.extractInstance(LabeledDatum.of("LABEL", "TeSt"));
      assertEquals("LABEL", ii.getLabel());
      assertEquals(1, ii.getFeatures().size());
      assertEquals(1.0, ii.getValue("TeSt"), 0.0);
      assertEquals(0.0, ii.getValue("test"), 0.0);
      assertEquals(0.0, ii.getValue("testing"), 0.0);


      chain = Featurizer.chain(f1,
                               ((Featurizer<String>) s -> Collections.singleton(Feature.TRUE(s.toLowerCase()))));
      ii = chain.extractInstance("TeSt", "LABEL");
      assertEquals("LABEL", ii.getLabel());
      assertEquals(2, ii.getFeatures().size());
      assertEquals(1.0, ii.getValue("TeSt"), 0.0);
      assertEquals(1.0, ii.getValue("test"), 0.0);
      assertEquals(0.0, ii.getValue("testing"), 0.0);
   }

   @Test
   public void asSequenceFeaturizer() throws Exception {
      SequenceFeaturizer<String> featurizer = f1.asSequenceFeaturizer();
      SequenceInput<String> input = new SequenceInput<>(Collections.singletonList("TeSt"));
      Sequence sequence = featurizer.extractSequence(input.iterator());
      assertEquals(1, sequence.size());
      Instance ii = sequence.asInstances().get(0);
      assertEquals(1, ii.getFeatures().size());
      assertEquals(1.0, ii.getValue("TeSt"), 0.0);
      assertEquals(0.0, ii.getValue("test"), 0.0);
      assertEquals(0.0, ii.getValue("testing"), 0.0);
   }


}