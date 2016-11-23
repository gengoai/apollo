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
import com.davidbracewell.apollo.ml.sequence.Context;
import com.davidbracewell.apollo.ml.sequence.SequenceFeaturizer;
import com.davidbracewell.string.StringUtils;
import com.google.common.base.Preconditions;
import lombok.NonNull;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import static com.davidbracewell.apollo.ml.sequence.Sequence.BOS;
import static com.davidbracewell.apollo.ml.sequence.Sequence.EOS;

/**
 * <p>Constructs NGram predicate features for all size n-grams (except unigrams) from <code>previousWindow</code>  to
 * <code>nextWindow</code> using the given {@link PredicateFeaturizer}</p>
 *
 * @param <E> the input type parameter
 * @author David B. Bracewell
 */
public class NGramSequenceFeaturizer<E> implements SequenceFeaturizer<E> {
   private static final long serialVersionUID = 1L;

   private final int previousWindow;
   private final int nextWindow;
   private final PredicateFeaturizer<? super E> featurizer;

   /**
    * Instantiates a new N gram sequence featurizer.
    *
    * @param previousWindow the previous window
    * @param nextWindow     the next window
    * @param featurizer     the featurizer
    */
   public NGramSequenceFeaturizer(int previousWindow, int nextWindow, @NonNull PredicateFeaturizer<? super E> featurizer) {
      Preconditions.checkState(previousWindow > 0 || nextWindow > 0, "Either previousWindow or nextWindow must be > 0");
      this.previousWindow = Math.abs(previousWindow);
      this.nextWindow = Math.abs(nextWindow);
      this.featurizer = featurizer;
   }


   @Override
   public Set<Feature> apply(Context<E> iterator) {
      final String prefix = featurizer.getPrefix();

      final String c0 = featurizer.extractPredicate(iterator.getCurrent());
      final int index = iterator.getIndex();

      List<String> previous = new ArrayList<>();
      for (int i = index - 1; i >= 0 && i >= index - previousWindow; i--) {
         iterator.getPrevious(i)
                 .ifPresent(v -> previous.add(featurizer.extractPredicate(v)));
      }
      if (previous.isEmpty()) {
         previous.add(BOS);
      }

      List<String> next = new ArrayList<>();
      for (int i = index + 1; iterator.getNext(i).isPresent() && i <= index + nextWindow; i++) {
         iterator.getNext(i)
                 .ifPresent(v -> next.add(featurizer.extractPredicate(v)));
      }
      if (next.isEmpty()) {
         next.add(EOS);
      }

      int zeroIndex = previous.size();
      List<String> all = new ArrayList<>(previous);
      all.add(c0);
      all.addAll(next);

      Set<Feature> features = new HashSet<>();

      for (int i = 0; i < all.size(); i++) {
         int start = i < zeroIndex ? -(zeroIndex - i) : i > zeroIndex + 1 ? (i - zeroIndex) : 0;
         for (int j = i + 1; j < all.size(); j++) {
            int end = j < zeroIndex ? -(zeroIndex - j) : j > zeroIndex + 1 ? (j - zeroIndex) : 0;
            features.add(Feature.TRUE(prefix + "[" + start + "..." + end + "]=" + StringUtils.join(all.subList(i,
                                                                                                               j + 1),
                                                                                                   "_")));
         }
      }

      return features;
   }
}//END OF NGramSequenceFeaturizer
