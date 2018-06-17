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
import com.gengoai.string.StringUtils;
import lombok.Getter;
import lombok.NonNull;

import java.util.Collections;
import java.util.List;

/**
 * <p>Specialized featurizer that produces one feature in the form <code>PREFIX=PREDICATE</code> allowing access to the
 * prefix and predicate separately. This is of most use for sequence labeling where features need positional elements
 * attached.</p>
 *
 * @param <INPUT> the input type parameter
 * @author David B. Bracewell
 */
public abstract class PredicateFeaturizer<INPUT> implements Featurizer<INPUT> {
   private static final long serialVersionUID = 1L;
   @Getter
   private final String prefix;

   /**
    * Instantiates a new Predicate featurizer.
    */
   protected PredicateFeaturizer(@NonNull String prefix) {
      this.prefix = prefix;
   }

   @Override
   public final List<Feature> apply(@NonNull INPUT input) {
      String predicate = extractPredicate(input);
      if (StringUtils.isNullOrBlank(predicate)) {
         return Collections.emptyList();
      }
      return Collections.singletonList(Feature.TRUE(prefix, predicate));
   }

   /**
    * Implementation to extract a single predicate from the given input.
    *
    * @param input the input
    * @return the predicate feature name
    */
   public abstract String extractPredicate(INPUT input);

}//END OF PredicateFeaturizer
