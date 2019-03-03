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

package com.gengoai.apollo.ml.regression;

import com.gengoai.Copyable;
import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.linear.NDArrayFactory;
import com.gengoai.apollo.linear.NDArrayInitializer;
import com.gengoai.apollo.ml.Example;
import com.gengoai.apollo.ml.NumericPipeline;
import com.gengoai.apollo.ml.data.Dataset;
import com.gengoai.apollo.ml.params.ParamMap;
import com.gengoai.apollo.ml.preprocess.Preprocessor;
import com.gengoai.apollo.optimization.*;
import com.gengoai.apollo.optimization.activation.Activation;
import com.gengoai.apollo.optimization.loss.SquaredLoss;

import java.io.Serializable;

/**
 * The type Linear regression.
 *
 * @author David B. Bracewell
 */
public class LinearRegression extends Regression {
   private static final long serialVersionUID = 1L;
   private final WeightParameters weightParameters = new WeightParameters();

   /**
    * Instantiates a new Linear regression.
    *
    * @param preprocessors the preprocessors
    */
   public LinearRegression(Preprocessor... preprocessors) {
      super(preprocessors);
   }

   /**
    * Instantiates a new Linear regression.
    *
    * @param modelParameters the model parameters
    */
   public LinearRegression(NumericPipeline modelParameters) {
      super(modelParameters);
   }

   @Override
   public double estimate(Example vector) {
      return weightParameters.activate(vector.preprocessAndTransform(getPipeline())).scalarValue();
   }

   @Override
   protected void fitPreprocessed(Dataset preprocessed, ParamMap p) {
      weightParameters.update(getNumberOfLabels(), getNumberOfFeatures());
      GradientDescentOptimizer optimizer = GradientDescentOptimizer.builder()
                                                                   .batchSize(p.get(batchSize)).build();

      optimizer.optimize(weightParameters,
                         () -> preprocessed.stream().map(e -> e.transform(getPipeline())),
                         new GradientDescentCostFunction(new SquaredLoss(), -1),
                         StoppingCriteria.create()
                                         .maxIterations(p.get(maxIterations))
                                         .historySize(p.get(historySize))
                                         .tolerance(p.get(tolerance)),
                         p.get(weightUpdater),
                         p.get(verbose) ? p.get(reportInterval) : -1);
   }

   @Override
   public ParamMap getDefaultFitParameters() {
      return new ParamMap(
         verbose.set(false),
         maxIterations.set(100),
         batchSize.set(32),
         historySize.set(3),
         tolerance.set(1e-5),
         reportInterval.set(50),
         cacheData.set(true),
         weightUpdater.set(SGDUpdater.builder().build())
      );
   }

   private static class WeightParameters implements LinearModelParameters, Serializable, Copyable<WeightParameters> {
      private static final long serialVersionUID = 1L;
      private Activation activation = Activation.SIGMOID;
      private NDArray bias;
      private int numFeatures;
      private int numLabels;
      private NDArray weights;

      @Override
      public WeightParameters copy() {
         return Copyable.deepCopy(this);
      }

      @Override
      public Activation getActivation() {
         return activation;
      }

      @Override
      public NDArray getBias() {
         return bias;
      }

      @Override
      public NDArray getWeights() {
         return weights;
      }

      @Override
      public int numberOfFeatures() {
         return numFeatures;
      }

      @Override
      public int numberOfLabels() {
         return numLabels;
      }

      /**
       * Update.
       *
       * @param numLabels   the num labels
       * @param numFeatures the num features
       */
      public void update(int numLabels, int numFeatures) {
         this.numLabels = numLabels;
         this.numFeatures = numFeatures;
         int numL = numLabels <= 2 ? 1 : numLabels;
         this.activation = Activation.LINEAR;
         this.weights = NDArrayFactory.DEFAULT().create(NDArrayInitializer.rand, numL, numFeatures);
         this.bias = NDArrayFactory.DEFAULT().zeros(numL);
      }
   }


}//END OF LinearRegression
