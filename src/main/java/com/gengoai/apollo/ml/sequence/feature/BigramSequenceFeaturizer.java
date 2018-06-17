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

package com.gengoai.apollo.ml.sequence.feature;

import com.gengoai.apollo.ml.Feature;
import com.gengoai.apollo.ml.featurizer.PredicateFeaturizer;
import com.gengoai.apollo.ml.sequence.Context;
import com.gengoai.apollo.ml.sequence.SequenceFeaturizer;
import com.gengoai.collection.Lists;
import lombok.NonNull;

import java.util.List;

import static com.gengoai.apollo.ml.sequence.Sequence.BOS;
import static com.gengoai.apollo.ml.sequence.Sequence.EOS;

/**
 * <p>Constructs Bigram predicate features with positions <code>[-1,0]</code> and <code>[0,+1]</code> using the given
 * {@link PredicateFeaturizer}</p>
 *
 * @param <E> the input type parameter
 * @author David B. Bracewell
 */
public class BigramSequenceFeaturizer<E> implements SequenceFeaturizer<E> {
   private static final long serialVersionUID = 1L;
   private final PredicateFeaturizer<? super E> featurizer;

   /**
    * Instantiates a new Bigram sequence featurizer.
    *
    * @param featurizer the predicate featurizer to use for extracting features
    */
   public BigramSequenceFeaturizer(@NonNull PredicateFeaturizer<? super E> featurizer) {
      this.featurizer = featurizer;
   }

   @Override
   public List<Feature> apply(Context<E> iterator) {
      final String c0 = featurizer.extractPredicate(iterator.getCurrent());
      final String p1 = iterator.getPrevious(1).map(featurizer::extractPredicate).orElse(BOS);
      final String n1 = iterator.getNext(1).map(featurizer::extractPredicate).orElse(EOS);
      return Lists.list(Feature.TRUE(featurizer.getPrefix() + "[-1,0]", p1, c0),
                        Feature.TRUE(featurizer.getPrefix() + "[0,+1]", c0, n1));
   }
}//END OF BigramSequenceFeaturizer
