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
 *
 */

package com.gengoai.apollo.ml.preprocess;

import com.gengoai.apollo.ml.Feature;
import com.gengoai.apollo.ml.Instance;
import com.gengoai.collection.Sets;
import com.gengoai.collection.counter.Counters;
import com.gengoai.stream.MStream;

import java.util.Collections;
import java.util.Optional;
import java.util.Set;

/**
 * @author David B. Bracewell
 */
public class TopNFilter extends RestrictedFeaturePreprocessor implements InstancePreprocessor {
   private static final long serialVersionUID = 1L;
   private volatile Set<String> selectedFeatures = Collections.emptySet();
   private int topN;
   private boolean trained = false;

   /**
    * Instantiates a new Top N filter.
    *
    * @param featurePrefix the feature prefix to restrict the filter to
    * @param topN          the number of features to keep
    */
   public TopNFilter(String featurePrefix, int topN) {
      super(featurePrefix);
      this.topN = topN;
   }

   /**
    * Instantiates a new Top N  filter with no restriction.
    *
    * @param topN the number of features to keep
    */
   public TopNFilter(int topN) {
      this(null, topN);
   }


   @Override
   public Instance applyInstance(Instance example) {
      if (trained) {
         return example;
      }
      return example.mapFeatures(f -> Optional.ofNullable(selectedFeatures.contains(f.name) ? f : null));
   }

   @Override
   protected void cleanup() {
      this.selectedFeatures = Collections.emptySet();
      this.trained = true;
   }

   @Override
   protected void fitFeatures(MStream<Feature> exampleStream) {
      selectedFeatures = Sets.asHashSet(Counters.newCounter(exampleStream.map(Feature::getName)
                                                                         .countByValue())
                                                .topN(topN)
                                                .items());
   }

   @Override
   public void reset() {
      this.selectedFeatures = Collections.emptySet();
      this.trained = false;
   }

   @Override
   public String toString() {
      return "TopNFilter[" + getRestriction() + "]{minCount=" + topN + "}";
   }

}//END OF TopNFilter
