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
import com.gengoai.math.EnhancedDoubleStatistics;
import com.gengoai.stream.MStream;

import java.util.Optional;

/**
 * <p>Transforms features values to Z-Scores.</p>
 *
 * @author David B. Bracewell
 */
public class ZScoreTransform extends RestrictedFeaturePreprocessor {
   private static final long serialVersionUID = 1L;

   private double mean = 0;
   private double standardDeviation = 0;

   /**
    * Instantiates a new Z score transform.
    */
   public ZScoreTransform() {
      super(null);
   }

   /**
    * Instantiates a new Z score transform.
    *
    * @param featureNamePrefix the feature name prefix
    */
   public ZScoreTransform(String featureNamePrefix) {
      super(featureNamePrefix);
   }

   @Override
   public Instance applyInstance(Instance example) {
      return example.mapFeatures(in -> {
         if (!requiresProcessing(in)) {
            return Optional.of(in);
         }
         if (standardDeviation == 0) {
            return Optional.of(Feature.realFeature(in.getName(), mean));
         }
         return Optional.of(Feature.realFeature(in.getName(), (in.getValue() - mean) / standardDeviation));
      });
   }

   @Override
   protected void fitFeatures(MStream<Feature> stream) {
      EnhancedDoubleStatistics stats = stream.mapToDouble(Feature::getValue).statistics();
      this.mean = stats.getAverage();
      this.standardDeviation = stats.getSampleStandardDeviation();
   }

   @Override
   public void reset() {
      mean = 0;
      standardDeviation = 0;
   }

   @Override
   public String toString() {
      return "ZScoreTransform[" + getRestriction() + "]{mean=" + mean + ", std=" + standardDeviation + "}";
   }

}//END OF ZScoreTransform
