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
import com.gengoai.apollo.ml.Example;
import com.gengoai.apollo.ml.FitParameters;
import com.gengoai.apollo.ml.ModelParameters;
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

import static com.gengoai.Validation.notNull;

/**
 * <p>A generalized linear model. This model can encompass a number models dependent on the parameters when
 * training.</p>
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
      super(ModelParameters.classification(false, p -> p.preprocessors(preprocessors)));
      this.weightParameters = new WeightParameters();
   }

   /**
    * Instantiates a new Linear model.
    *
    * @param modelParameters the model parameters
    */
   public LinearModel(ModelParameters modelParameters) {
      super(modelParameters);
      this.weightParameters = new WeightParameters();
   }


   @Override
   protected Classifier fitPreprocessed(Dataset preprocessed, FitParameters fitParameters) {
      Parameters parameters = notNull(Cast.as(fitParameters, Parameters.class));
      this.weightParameters.numFeatures = getNumberOfFeatures();
      this.weightParameters.numLabels = getNumberOfLabels();
      GradientDescentOptimizer optimizer = GradientDescentOptimizer.builder()
                                                                   .batchSize(parameters.batchSize).build();

      final SerializableSupplier<MStream<NDArray>> dataSupplier;
      if (parameters.cacheData) {
         if (parameters.verbose) {
            logInfo("Caching dataset...");
         }
         final MStream<NDArray> cached = preprocessed.stream()
                                                     .map(this::encode)
                                                     .cache();
         dataSupplier = () -> cached;
      } else {
         dataSupplier = () -> preprocessed.stream()
                                          .map(this::encode);
      }
      this.weightParameters.update(parameters);
      optimizer.optimize(weightParameters,
                         dataSupplier,
                         new GradientDescentCostFunction(parameters.lossFunction, -1),
                         TerminationCriteria.create()
                                            .maxIterations(parameters.maxIterations)
                                            .historySize(parameters.historySize)
                                            .tolerance(parameters.tolerance),
                         parameters.weightUpdater,
                         parameters.verbose ? parameters.reportInterval
                                            : -1);

      return this;
   }

   @Override
   public Parameters getDefaultFitParameters() {
      return new Parameters();
   }

   @Override
   public Classification predict(Example example) {
      return new Classification(weightParameters.activate(encodeAndPreprocess(example)), getLabelVectorizer());
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
         this.activation = parameters.activation;
         this.weights = NDArrayFactory.DEFAULT().create(NDArrayInitializer.rand, numL, numFeatures);
         this.bias = NDArrayFactory.DEFAULT().zeros(numL);
      }
   }

   /**
    * Custom fit parameters for the LinearModel
    */
   public static class Parameters extends com.gengoai.apollo.ml.FitParameters {
      private static final long serialVersionUID = 1L;
      /**
       * The Activation.
       */
      public Activation activation = Activation.SIGMOID;
      /**
       * The Batch size.
       */
      public int batchSize = 20;
      /**
       * The Cache data.
       */
      public boolean cacheData = true;
      /**
       * The History size.
       */
      public int historySize = 3;
      /**
       * The Loss function.
       */
      public LossFunction lossFunction = new LogLoss();
      /**
       * The Max iterations.
       */
      public int maxIterations = 300;
      /**
       * The Report interval.
       */
      public int reportInterval = 100;
      /**
       * The Tolerance.
       */
      public double tolerance = 1e-9;
      /**
       * The Weight updater.
       */
      public WeightUpdate weightUpdater = SGDUpdater.builder().build();

      @Override
      public String toString() {
         return "Parameters{" +
                   "activation=" + activation +
                   ", batchSize=" + batchSize +
                   ", cacheData=" + cacheData +
                   ", historySize=" + historySize +
                   ", lossFunction=" + lossFunction +
                   ", maxIterations=" + maxIterations +
                   ", tolerance=" + tolerance +
                   ", weightUpdater=" + weightUpdater +
                   ", verbose=" + verbose +
                   ", reportInterval=" + reportInterval +
                   '}';
      }
   }


}//END OF LinearModel
