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

package com.davidbracewell.apollo.ml.sequence.feature;

import com.davidbracewell.apollo.ml.Feature;
import com.davidbracewell.apollo.ml.PredicateFeaturizer;
import com.davidbracewell.apollo.ml.sequence.ContextualIterator;
import com.davidbracewell.apollo.ml.sequence.SequenceFeaturizer;
import lombok.NonNull;

import java.util.HashSet;
import java.util.Set;

import static com.davidbracewell.apollo.ml.sequence.Sequence.BOS;
import static com.davidbracewell.apollo.ml.sequence.Sequence.EOS;

/**
 * @author David B. Bracewell
 */
public class WindowedSequenceFeaturizer<E> implements SequenceFeaturizer<E> {
   private static final long serialVersionUID = 1L;

   private final int previousWindow;
   private final int nextWindow;
   private final PredicateFeaturizer<? super E> featurizer;
   private final boolean includeSequenceBoundaries;

   public WindowedSequenceFeaturizer(int previousWindow, int nextWindow, @NonNull PredicateFeaturizer<? super E> featurizer) {
      this(previousWindow, nextWindow, false, featurizer);
   }

   public WindowedSequenceFeaturizer(int previousWindow, int nextWindow, boolean includeSequenceBoundaries, @NonNull PredicateFeaturizer<? super E> featurizer) {
      this.previousWindow = Math.abs(previousWindow);
      this.nextWindow = Math.abs(nextWindow);
      this.featurizer = featurizer;
      this.includeSequenceBoundaries = includeSequenceBoundaries;
   }


   @Override
   public Set<Feature> apply(ContextualIterator<E> iterator) {
      Set<Feature> features = new HashSet<>();
      final String prefix = featurizer.getPrefix();

      features.add(Feature.TRUE(prefix + "[0]", featurizer.extractPredicate(iterator.getCurrent())));

      //Add features for previous observation
      for (int i = 1; i <= previousWindow && (i == 1 || iterator.getPrevious(i).isPresent()); i++) {
         String position = "[-" + i + "]=";
         String predicate = iterator.getPrevious(i).map(featurizer::extractPredicate).orElse(includeSequenceBoundaries
                                                                                             ? BOS
                                                                                             : null);
         if (predicate != null) {
            features.add(Feature.TRUE(featurizer.getPrefix() + position, predicate));
         }
      }

      //Add all features for the next observation(s)
      for (int i = 1; i <= nextWindow && (i == 1 || iterator.getNext(i).isPresent()); i++) {
         String position = "[+" + i + "]=";
         String predicate = iterator.getNext(i).map(featurizer::extractPredicate).orElse(includeSequenceBoundaries
                                                                                         ? EOS
                                                                                         : null);
         if (predicate != null) {
            features.add(Feature.TRUE(featurizer.getPrefix() + position, predicate));
         }
      }

      return features;
   }
}//END OF WindowedFeaturizer
