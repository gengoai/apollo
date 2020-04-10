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
import com.gengoai.LogUtils;
import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.ml.Example;
import com.gengoai.apollo.ml.FitParameters;
import com.gengoai.apollo.ml.NumericPipeline;
import com.gengoai.apollo.ml.Params;
import com.gengoai.apollo.ml.data.ExampleDataset;
import com.gengoai.apollo.ml.data.VectorizedDataset;
import com.gengoai.apollo.ml.preprocess.Preprocessor;
import com.gengoai.apollo.optimization.*;
import com.gengoai.apollo.optimization.activation.Activation;
import com.gengoai.apollo.optimization.loss.SquaredLoss;
import com.gengoai.conversion.Cast;
import lombok.NonNull;

import java.io.Serializable;

import static com.gengoai.Validation.notNull;
import static com.gengoai.apollo.linear.NDArrayFactory.ND;

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
      return weightParameters.activate(vector.preprocessAndTransform(getPipeline())).scalar();
   }

   @Override
   public void fit(@NonNull VectorizedDataset dataset, @NonNull FitParameters<?> fitParameters) {
      Parameters p = notNull(Cast.as(fitParameters, Parameters.class));
      weightParameters.update(getNumberOfLabels(), getNumberOfFeatures());
      GradientDescentOptimizer optimizer = new GradientDescentOptimizer(p.batchSize.value());
      optimizer.optimize(weightParameters,
                         dataset.stream(),
                         new GradientDescentCostFunction(new SquaredLoss(), -1),
                         StoppingCriteria.create("loss", p)
                                         .historySize(p.historySize.value())
                                         .reportInterval(p.reportInterval.value())
                                         .logger(LogUtils.getLogger(LinearRegression.class)),
                         p.weightUpdater.value());
   }

   @Override
   protected void fitPreprocessed(@NonNull ExampleDataset preprocessed, @NonNull FitParameters<?> fitParameters) {
      fit(preprocessed.toVectorizedDataset(getPipeline()), fitParameters);
   }

   @Override
   public LinearRegression.Parameters getFitParameters() {
      return new Parameters();
   }

   /**
    * The type Parameters.
    */
   public static class Parameters extends FitParameters<Parameters> {
      private static final long serialVersionUID = 1L;
      public final Parameter<Integer> batchSize = parameter(Params.Optimizable.batchSize, 20);
      public final Parameter<Integer> historySize = parameter(Params.Optimizable.historySize, 3);
      public final Parameter<Integer> maxIterations = parameter(Params.Optimizable.maxIterations, 100);
      public final Parameter<Integer> reportInterval = parameter(Params.Optimizable.reportInterval, 100);
      public final Parameter<Double> tolerance = parameter(Params.Optimizable.tolerance, 1e-9);
      public final Parameter<WeightUpdate> weightUpdater = parameter(Params.Optimizable.weightUpdate,
                                                                     SGDUpdater.builder().build());

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
         int numL = numLabels <= 2
                    ? 1
                    : numLabels;
         this.activation = Activation.LINEAR;
         this.weights = ND.rand(numL, numFeatures);
         this.bias = ND.array(numL);
      }
   }

}//END OF LinearRegression
