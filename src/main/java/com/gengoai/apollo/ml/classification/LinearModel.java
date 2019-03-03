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
import com.gengoai.apollo.ml.Model;
import com.gengoai.apollo.ml.data.Dataset;
import com.gengoai.apollo.ml.params.ParamMap;
import com.gengoai.apollo.ml.preprocess.Preprocessor;
import com.gengoai.apollo.optimization.*;
import com.gengoai.apollo.optimization.activation.Activation;
import com.gengoai.apollo.optimization.loss.LogLoss;
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
   protected void fitPreprocessed(Dataset preprocessed, ParamMap fitParameters) {
      this.weightParameters.numFeatures = getNumberOfFeatures();
      this.weightParameters.numLabels = getNumberOfLabels();
      GradientDescentOptimizer optimizer = GradientDescentOptimizer.builder()
                                                                   .batchSize(fitParameters.get(batchSize)).build();

      final SerializableSupplier<MStream<NDArray>> dataSupplier;
      if (fitParameters.get(cacheData)) {
         if (fitParameters.get(verbose)) {
            logInfo("Caching dataset...");
         }
         final MStream<NDArray> cached = preprocessed.asVectorStream(getPipeline()).cache();
         dataSupplier = () -> cached;
      } else {
         dataSupplier = () -> preprocessed.asVectorStream(getPipeline());
      }
      this.weightParameters.update(fitParameters);
      optimizer.optimize(weightParameters,
                         dataSupplier,
                         new GradientDescentCostFunction(fitParameters.get(lossFunction), -1),
                         StoppingCriteria.create()
                                         .maxIterations(fitParameters.get(maxIterations))
                                         .historySize(fitParameters.get(historySize))
                                         .tolerance(fitParameters.get(tolerance)),
                         fitParameters.get(weightUpdater),
                         fitParameters.get(verbose) ? fitParameters.get(reportInterval) : -1);
   }

   @Override
   public Parameters getFitParameters() {
      return new Parameters();
   }

   public static class Parameters extends ParamMap {

      public Parameters() {
         super(verbose.set(false),
               maxIterations.set(100),
               activation.set(Activation.SIGMOID),
               batchSize.set(32),
               historySize.set(3),
               tolerance.set(1e-5),
               reportInterval.set(50),
               cacheData.set(true),
               lossFunction.set(new LogLoss()),
               weightUpdater.set(SGDUpdater.builder().build()));
      }

      public Parameters verbose(boolean isVerbose) {
         update(verbose.set(isVerbose));
         return this;
      }

      public Parameters maxIterations(int maxIterations) {
         update(Model.maxIterations.set(maxIterations));
         return this;
      }

   }

   @Override
   public Classification predict(Example example) {
      return new Classification(weightParameters.activate(example.preprocessAndTransform(getPipeline())),
                                getPipeline().labelVectorizer);
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
      public void update(ParamMap parameters) {
         int numL = numLabels <= 2 ? 1 : numLabels;
         this.activation = parameters.get(Model.activation);
         this.weights = NDArrayFactory.DEFAULT().create(NDArrayInitializer.rand, numL, numFeatures);
         this.bias = NDArrayFactory.DEFAULT().zeros(numL);
      }
   }


}//END OF LinearModel
