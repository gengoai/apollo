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

import com.davidbracewell.string.StringUtils;
import lombok.Getter;
import lombok.NonNull;

import java.util.Collections;
import java.util.Set;

/**
 * The type Predicate featurizer.
 *
 * @param <INPUT> the type parameter
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
   public final Set<Feature> apply(@NonNull INPUT input) {
      String predicate = extractPredicate(input);
      if (StringUtils.isNullOrBlank(predicate)) {
         return Collections.emptySet();
      }
      return Collections.singleton(Feature.TRUE(prefix, predicate));
   }

   /**
    * Extract predicate string.
    *
    * @param input the input
    * @return the string
    */
   public abstract String extractPredicate(INPUT input);

}//END OF PredicateFeaturizer