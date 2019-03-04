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

package com.gengoai.apollo.ml.classification;

import com.gengoai.Copyable;
import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.linear.NDArrayFactory;
import com.gengoai.apollo.linear.NDArrayInitializer;
import com.gengoai.apollo.ml.DiscretePipeline;
import com.gengoai.apollo.ml.Example;
import com.gengoai.apollo.ml.FitParameters;
import com.gengoai.apollo.ml.Params;
import com.gengoai.apollo.ml.data.Dataset;
import com.gengoai.apollo.ml.preprocess.Preprocessor;
import com.gengoai.apollo.optimization.*;
import com.gengoai.apollo.optimization.activation.Activation;
import com.gengoai.apollo.optimization.loss.LogLoss;
import com.gengoai.apollo.optimization.loss.LossFunction;
import com.gengoai.conversion.Cast;
import com.gengoai.function.SerializableSupplier;
import com.gengoai.logging.Loggable;
import com.gengoai.stream.MStream;

import java.io.Serializable;

/**
 * <p>A generalized linear model. This model can encompass a number models dependent on the parameters when
 * training. A linear model learns a function to estimate observations (y) from one or more independent variables (x) in
 * the form of y = b_0 + x_1 * b_1 + x_2 * b_2 ... + x_n * b_n, where b_i represents the weight of the corresponding
 * variable x_i.  The LinearModel class allows for training Logistic Regression and SVM classifiers.</p>
 *
 * @author David B. Bracewell
 */
public class LinearModel extends Classifier implements Loggable {
   private static final long serialVersionUID = 1L;
   private final WeightParameters weightParameters;

   /**
    * Instantiates a new Linear model.
    *
    * @param preprocessors the preprocessors
    */
   public LinearModel(Preprocessor... preprocessors) {
      super(preprocessors);
      this.weightParameters = new WeightParameters();
   }

   /**
    * Instantiates a new Linear model.
    *
    * @param modelParameters the model parameters
    */
   public LinearModel(DiscretePipeline modelParameters) {
      super(modelParameters);
      this.weightParameters = new WeightParameters();
   }


   @Override
   protected void fitPreprocessed(Dataset preprocessed, FitParameters fitParameters) {
      Parameters parameters = Cast.as(fitParameters);
      this.weightParameters.numFeatures = getNumberOfFeatures();
      this.weightParameters.numLabels = getNumberOfLabels();
      GradientDescentOptimizer optimizer = GradientDescentOptimizer.builder()
                                                                   .batchSize(parameters.batchSize.value()).build();

      final SerializableSupplier<MStream<NDArray>> dataSupplier;
      if (parameters.cacheData.value()) {
         if (parameters.verbose.value()) {
            logInfo("Caching dataset...");
         }
         final MStream<NDArray> cached = preprocessed.asVectorStream(getPipeline()).cache();
         dataSupplier = () -> cached;
      } else {
         dataSupplier = () -> preprocessed.asVectorStream(getPipeline());
      }
      this.weightParameters.update(parameters);
      optimizer.optimize(weightParameters,
                         dataSupplier,
                         new GradientDescentCostFunction(parameters.lossFunction.value(), -1),
                         StoppingCriteria.create()
                                         .maxIterations(parameters.maxIterations.value())
                                         .historySize(parameters.historySize.value())
                                         .tolerance(parameters.tolerance.value()),
                         parameters.weightUpdater.value(),
                         parameters.verbose.value() ? parameters.reportInterval.value()
                                                    : -1);
   }

   @Override
   public Parameters getFitParameters() {
      return new Parameters();
   }

   @Override
   public Classification predict(Example example) {
      return new Classification(weightParameters.activate(example.preprocessAndTransform(getPipeline())),
                                getPipeline().labelVectorizer);
   }

   /**
    * Custom fit parameters for the LinearModel
    */
   public static class Parameters extends FitParameters<Parameters> {
      private static final long serialVersionUID = 1L;
      public final Parameter<Activation> activation = parameter(Params.Optimizable.activation, Activation.SIGMOID);
      public final Parameter<Integer> batchSize = parameter(Params.Optimizable.batchSize, 32);
      public final Parameter<Integer> historySize = parameter(Params.Optimizable.historySize, 3);
      public final Parameter<Integer> maxIterations = parameter(Params.Optimizable.maxIterations, 100);
      public final Parameter<Integer> reportInterval = parameter(Params.Optimizable.reportInterval, 100);
      public final Parameter<Double> tolerance = parameter(Params.Optimizable.tolerance, 1e-9);
      public final Parameter<LossFunction> lossFunction = parameter(Params.Optimizable.lossFunction, new LogLoss());
      public final Parameter<Boolean> cacheData = parameter(Params.Optimizable.cacheData, true);
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
       * @param parameters the fit parameters
       */
      public void update(Parameters parameters) {
         int numL = numLabels <= 2 ? 1 : numLabels;
         this.activation = parameters.activation.value();
         this.weights = NDArrayFactory.DEFAULT().create(NDArrayInitializer.rand, numL, numFeatures);
         this.bias = NDArrayFactory.DEFAULT().zeros(numL);
      }
   }


}//END OF LinearModel
