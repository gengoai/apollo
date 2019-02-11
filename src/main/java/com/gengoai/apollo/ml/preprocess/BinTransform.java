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
import com.gengoai.stream.MStream;
import com.gengoai.stream.accumulator.MStatisticsAccumulator;

import java.util.Arrays;
import java.util.Optional;

/**
 * <p>Converts a real valued feature into a number of binary features by creating a <code>bin</code> number of new
 * binary features. Creates <code>bin</code> number of equal sized bins that the feature values can fall into.</p>
 *
 * @author David B. Bracewell
 */
public class BinTransform extends RestrictedFeaturePreprocessor implements InstancePreprocessor {
   private static final long serialVersionUID = 1L;
   private double[] bins;

   /**
    * Instantiates a new BinTransform with no restriction
    *
    * @param numberOfBins the number of bins to convert the feature into
    */
   public BinTransform(int numberOfBins) {
      this(null, numberOfBins);
   }

   /**
    * Instantiates a new BinTransform.
    *
    * @param featureNamePrefix the feature name prefix to restrict to
    * @param numberOfBins      the number of bins to convert the feature into
    */
   public BinTransform(String featureNamePrefix, int numberOfBins) {
      super(featureNamePrefix);
      Validation.checkArgument(numberOfBins > 0, "Number of bins must be > 0.");
      this.bins = new double[numberOfBins];
   }

   /**
    * Instantiates a new Real to discrete transform.
    */
   protected BinTransform() {
      this(null, 1);
   }


   @Override
   public Instance applyInstance(Instance example) {
      return example.mapFeatures(f -> {
         if (requiresProcessing(f)) {
            int bin = 0;
            for (; bin < bins.length - 1; bin++) {
               if (f.getValue() < bins[bin]) {
                  break;
               }
            }
            return Optional.of(Feature.booleanFeature(f.getPrefix(), "Bin[" + bin + "]"));
         }
         return Optional.of(f);
      });
   }

   @Override
   protected void fitFeatures(MStream<Feature> stream) {
      MStatisticsAccumulator stats = stream.getContext().statisticsAccumulator();
      stream.forEach(feature -> stats.add(feature.getValue()));
      EnhancedDoubleStatistics statistics = stats.value();
      double max = statistics.getMax();
      double min = statistics.getMin();
      double binSize = ((max - min) / bins.length);
      double sum = min;
      for (int i = 0; i < bins.length; i++) {
         sum += binSize;
         bins[i] = sum;
      }
   }

   @Override
   public void reset() {
      if (bins != null) {
         Arrays.fill(bins, 0);
      }
   }


   @Override
   public String toString() {
      return "BinTransform[" + getRestriction() + "]{numberOfBins=" + bins.length + "}";
   }

}//END OF BinTransform
