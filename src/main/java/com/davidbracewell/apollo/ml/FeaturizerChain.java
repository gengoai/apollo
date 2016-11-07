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

import com.davidbracewell.conversion.Cast;
import lombok.NonNull;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * @author David B. Bracewell
 */
class FeaturizerChain<INPUT> implements Featurizer<INPUT> {
   private static final long serialVersionUID = 1L;
   private final Set<Featurizer<INPUT>> featurizers = new HashSet<>();


   @SafeVarargs
   public FeaturizerChain(Featurizer<? super INPUT>... featurizers) {
      this.featurizers.addAll(Cast.cast(Arrays.asList(featurizers)));
   }


   public void addFeaturizer(Featurizer<? super INPUT> featurizer) {
      this.featurizers.add(Cast.as(featurizer));
   }

   @Override
   public Set<Feature> apply(@NonNull INPUT input) {
      return featurizers.parallelStream()
                        .flatMap(f -> f.apply(input).stream())
                        .collect(Collectors.toSet());
   }

}//END OF FeaturizerChain
