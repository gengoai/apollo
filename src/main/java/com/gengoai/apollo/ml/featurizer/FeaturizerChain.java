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

package com.gengoai.apollo.ml.featurizer;

import com.gengoai.apollo.ml.Feature;
import com.gengoai.conversion.Cast;
import lombok.NonNull;

import java.util.Collection;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * Represents a set of featurizers to run on a given input
 *
 * @param <INPUT> the input type parameter
 * @author David B. Bracewell
 */
class FeaturizerChain<INPUT> implements Featurizer<INPUT> {
   private static final long serialVersionUID = 1L;
   private final Set<Featurizer<? super INPUT>> featurizers = new HashSet<>();


   /**
    * Instantiates a new Featurizer chain.
    *
    * @param featurizerOne the first featurizer
    * @param featurizers   the featurizers making up the chain
    */
   public FeaturizerChain(@NonNull Featurizer<? super INPUT> featurizerOne,
                          Collection<Featurizer<? super INPUT>> featurizers) {
      this.featurizers.add(Cast.as(featurizerOne));
      if (featurizers != null && featurizers.size() > 0) {
         this.featurizers.addAll(featurizers);
      }
   }


   /**
    * Adds a featurizer to the chain.
    *
    * @param featurizer the featurizer
    */
   public void addFeaturizer(Featurizer<? super INPUT> featurizer) {
      this.featurizers.add(Cast.as(featurizer));
   }

   @Override
   public List<Feature> apply(@NonNull INPUT input) {
      return featurizers.parallelStream()
                        .flatMap(f -> f.apply(input).stream())
                        .collect(Collectors.toList());
   }

}//END OF FeaturizerChain
