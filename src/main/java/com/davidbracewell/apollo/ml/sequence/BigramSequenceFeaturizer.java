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

package com.davidbracewell.apollo.ml.sequence;

import com.davidbracewell.apollo.ml.Feature;
import com.davidbracewell.apollo.ml.Featurizer;
import lombok.NonNull;

import java.util.Collections;
import java.util.HashSet;
import java.util.Set;

import static com.davidbracewell.apollo.ml.sequence.Sequence.BOS;
import static com.davidbracewell.apollo.ml.sequence.Sequence.EOS;
import static com.davidbracewell.collection.Streams.zip;

/**
 * @author David B. Bracewell
 */
public class BigramSequenceFeaturizer<E> implements SequenceFeaturizer<E> {
   private static final long serialVersionUID = 1L;
   public static final String PREVIOUS_PREFIX = "P";
   public static final String NEXT_PREFIX = "N";

   private final Featurizer<? super E> featurizer;

   public BigramSequenceFeaturizer(@NonNull Featurizer<? super E> featurizer) {
      this.featurizer = featurizer;
   }


   @Override
   public Set<Feature> apply(ContextualIterator<E> iterator) {
      Set<Feature> features = new HashSet<>();

      Set<Feature> c0 = featurizer.apply(iterator.getCurrent());

      Set<Feature> p1 = iterator.getPrevious(1).map(featurizer::apply).orElse(Collections.singleton(Feature.TRUE(BOS)));
      zip(p1.stream(), c0.stream())
            .forEach(e -> features.add(Feature.TRUE(PREVIOUS_PREFIX + "1::" + e.getKey().getName() + "::" + e.getValue()
                                                                                                             .getName())));

      Set<Feature> n1 = iterator.getNext(1).map(featurizer::apply).orElse(Collections.singleton(Feature.TRUE(EOS)));
      zip(c0.stream(), n1.stream())
            .forEach(e -> features.add(Feature.TRUE(e.getKey().getName() + "::" + NEXT_PREFIX + "1::" + e.getValue()
                                                                                                         .getName())));

      return features;
   }
}//END OF WindowedFeaturizer
