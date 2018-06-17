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

package com.gengoai.apollo.ml.sequence;

import com.gengoai.apollo.ml.Feature;
import com.gengoai.apollo.ml.Instance;
import com.gengoai.apollo.ml.classification.LibLinearModel;
import com.gengoai.collection.Lists;
import lombok.NonNull;

import java.util.Iterator;
import java.util.List;

/**
 * <p>Implementation of Maximum Entropy Markov Model using LibLinear</p>
 *
 * @author David B. Bracewell
 */
public class MEMM extends SequenceLabeler {
   private static final long serialVersionUID = 1L;
   /**
    * The Model.
    */
   LibLinearModel model;


   /**
    * Instantiates a new MEMM.
    *
    * @param learner the learner
    */
   public MEMM(@NonNull MEMMLearner learner) {
      super(learner);
   }

   @Override
   public double[] estimate(Iterator<Feature> observation, Iterator<String> transitions) {
      List<Feature> features = Lists.asArrayList(observation);
      while (transitions.hasNext()) {
         features.add(Feature.TRUE(transitions.next()));
      }
      return model.classify(Instance.create(features)).distribution();
   }

}// END OF MEMM
