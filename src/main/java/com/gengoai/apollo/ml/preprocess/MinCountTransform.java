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
import com.gengoai.stream.MStream;
import com.gengoai.string.Strings;

import java.util.HashSet;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

/**
 * /**
 * <p>Removes features that occur in less than a given number of instances. Optionally, transforming them into an a
 * special feature with a name relating to "unknown" or "out-of-vocabulary".</p>
 *
 * @author David B. Bracewell
 */
public class MinCountTransform extends RestrictedFeaturePreprocessor {
   private static final long serialVersionUID = 1L;
   private final Set<String> toKeep = new HashSet<>();
   private int minCount;
   private String unknownFeature;
   private boolean trained = false;

   /**
    * Instantiates a new Min count.
    *
    * @param minCount the min count
    */
   public MinCountTransform(int minCount) {
      this(null, minCount, null);
   }

   /**
    * Instantiates a new Min count.
    *
    * @param minCount       the min count
    * @param unknownFeature the unknown feature
    */
   public MinCountTransform(int minCount, String unknownFeature) {
      this(null, minCount, unknownFeature);
   }


   /**
    * Instantiates a new Min count.
    *
    * @param featureNamePrefix the feature name prefix
    * @param minCount          the min count
    */
   public MinCountTransform(String featureNamePrefix, int minCount) {
      this(featureNamePrefix, minCount, null);
   }

   /**
    * Instantiates a new Min count.
    *
    * @param featureNamePrefix the feature name prefix
    * @param minCount          the min count
    * @param unknownFeature    the unknown feature
    */
   public MinCountTransform(String featureNamePrefix, int minCount, String unknownFeature) {
      super(featureNamePrefix);
      this.minCount = minCount;
      if (Strings.isNotNullOrBlank(unknownFeature)) {
         this.unknownFeature = (Strings.isNotNullOrBlank(featureNamePrefix)
                                ? featureNamePrefix
                                : Strings.EMPTY) + unknownFeature;
      }
   }

   @Override
   public Instance applyInstance(Instance example) {
      if (trained) {
         return example;
      }
      return example.mapFeatures(in -> {
         if (!requiresProcessing(in)) {
            return Optional.of(in);
         }
         if (toKeep.contains(in.getName())) {
            return Optional.of(in);
         } else if (!Strings.isNullOrBlank(unknownFeature)) {
            return Optional.of(Feature.realFeature(unknownFeature, in.getValue()));
         }
         return Optional.empty();
      });
   }

   @Override
   protected void fitFeatures(MStream<Feature> exampleStream) {
      Map<String, Long> m = exampleStream.map(Feature::getName).countByValue();
      m.values().removeIf(v -> v < minCount);
      this.toKeep.addAll(m.keySet());
   }

   @Override
   public void reset() {
      toKeep.clear();
      trained = false;
   }

   @Override
   public String toString() {
      return "MinCount[" + getRestriction() + "]{minCount=" + minCount + "}";
   }

   @Override
   public void cleanup() {
      if (Strings.isNullOrBlank(unknownFeature)) {
         toKeep.clear();
      }
      trained = true;
   }

}//END OF MinCountTransform
