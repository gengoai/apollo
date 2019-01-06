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

import com.gengoai.Validation;
import com.gengoai.apollo.ml.Feature;
import com.gengoai.apollo.ml.Instance;
import com.gengoai.math.EnhancedDoubleStatistics;
import com.gengoai.math.Math2;
import com.gengoai.stream.MStream;

import java.util.Optional;

/**
 * <p>Transforms features values to a new minimum and maximum based on the current minimum and maximum of the values in
 * the dataset.</p>
 *
 * @author David B. Bracewell
 */
public class RescaleTransform extends RestrictedFeaturePreprocessor {
   private static final long serialVersionUID = 1L;
   private final double newMax;
   private final double newMin;
   private double max = Double.NEGATIVE_INFINITY;
   private double min = Double.POSITIVE_INFINITY;


   /**
    * Instantiates a new Rescale transform.
    *
    * @param featureNamePrefix the feature name prefix
    */
   public RescaleTransform(String featureNamePrefix) {
      this(featureNamePrefix, 0.0, 1.0);
   }

   /**
    * Instantiates a new Rescale transform.
    */
   public RescaleTransform() {
      this(null, 0.0, 1.0);
   }

   /**
    * Instantiates a new Z score transform.
    *
    * @param newMin the new min
    * @param newMax the new max
    */
   public RescaleTransform(double newMin, double newMax) {
      this(null, newMin, newMax);
   }

   /**
    * Instantiates a new Z score transform.
    *
    * @param featureNamePrefix the feature name prefix
    * @param newMin            the new min
    * @param newMax            the new max
    */
   public RescaleTransform(String featureNamePrefix, double newMin, double newMax) {
      super(featureNamePrefix);
      Validation.checkArgument(newMin < newMax, "Min must be less than max");
      this.newMin = newMin;
      this.newMax = newMax;
   }

   @Override
   protected void fitFeatures(MStream<Feature> stream) {
      EnhancedDoubleStatistics stats = stream.mapToDouble(f -> f.value).statistics();
      this.min = stats.getMin();
      this.max = stats.getMax();
   }

   @Override
   public Instance applyInstance(Instance example) {
      return example.mapFeatures(in -> {
         if (!requiresProcessing(in)) {
            return Optional.of(in);
         }
         if (min == max) {
            return Optional.of(Feature.realFeature(in.name, newMax));
         }
         return Optional.of(Feature.realFeature(in.name, Math2.rescale(in.value, min, max, newMin, newMax)));
      });
   }

   @Override
   public void reset() {
      max = Double.NEGATIVE_INFINITY;
      min = Double.POSITIVE_INFINITY;
   }

   @Override
   public String toString() {
      return "ZScoreTransform[" + getRestriction() + "]" +
                "{sample_min=" + min + ", sample_max=" + max +
                ", target_min=" + newMin + ", target_max=" + newMax + "}";
   }
}//END OF RescaleTransform
