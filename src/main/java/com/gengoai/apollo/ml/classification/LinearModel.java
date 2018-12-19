package com.gengoai.apollo.ml.classification;

import com.gengoai.Copyable;
import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.linear.NDArrayFactory;
import com.gengoai.apollo.linear.NDArrayInitializer;
import com.gengoai.apollo.ml.FitParameters;
import com.gengoai.apollo.optimization.*;
import com.gengoai.apollo.optimization.activation.Activation;
import com.gengoai.apollo.optimization.loss.LogLoss;
import com.gengoai.apollo.optimization.loss.LossFunction;
import com.gengoai.conversion.Cast;
import com.gengoai.function.SerializableSupplier;
import com.gengoai.io.resource.ByteArrayResource;
import com.gengoai.logging.Loggable;
import com.gengoai.stream.MStream;

import java.io.Serializable;

/**
 * <p>A generalized linear model.</p>
 *
 * @author David B. Bracewell
 */
public class LinearModel implements Classifier, Loggable {
   private static final long serialVersionUID = 1L;
   private final ModelParameters modelParameters;

   /**
    * Instantiates a new Linear model.
    */
   public LinearModel() {
      this.modelParameters = new ModelParameters();
   }


   @Override
   public NDArray estimate(NDArray example) {
      return example.setPredicted(modelParameters.activate(example));
   }

   /**
    * Custom fit method taking LinearModel fit parameters object.
    *
    * @param dataSupplier the data supplier
    * @param parameters   the parameters
    */
   public void fit(SerializableSupplier<MStream<NDArray>> dataSupplier, Parameters parameters) {
      GradientDescentOptimizer optimizer = GradientDescentOptimizer.builder()
                                                                   .batchSize(parameters.batchSize).build();
      if (parameters.cacheData) {
         if (parameters.verbose) {
            logInfo("Caching dataset...");
         }
         MStream<NDArray> cached = dataSupplier.get().cache();
         dataSupplier = () -> cached;
      }
      this.modelParameters.update(parameters);
      optimizer.optimize(modelParameters,
                         dataSupplier,
                         new GradientDescentCostFunction(parameters.lossFunction, -1),
                         TerminationCriteria.create()
                                            .maxIterations(parameters.maxIterations)
                                            .historySize(parameters.historySize)
                                            .tolerance(parameters.tolerance),
                         parameters.weightUpdater,
                         parameters.verbose ? parameters.reportInterval
                                            : -1);
   }

   @Override
   public void fit(SerializableSupplier<MStream<NDArray>> dataSupplier, FitParameters fitParameters) {
      fit(dataSupplier, Cast.as(fitParameters, Parameters.class));
   }

   @Override
   public Parameters getDefaultFitParameters() {
      return new Parameters();
   }

   @Override
   public int getNumberOfFeatures() {
      return modelParameters.numFeatures;
   }

   @Override
   public int getNumberOfLabels() {
      return modelParameters.numLabels;
   }

   private static class ModelParameters implements LinearModelParameters, Serializable, Copyable<ModelParameters> {
      private static final long serialVersionUID = 1L;
      private Activation activation = Activation.SIGMOID;
      private NDArray bias;
      private int numFeatures;
      private int numLabels;
      private NDArray weights;

      @Override
      public ModelParameters copy() {
         try {
            return new ByteArrayResource().writeObject(this).readObject();
         } catch (Exception e) {
            throw new RuntimeException(e);
         }
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
         this.numLabels = parameters.numLabels;
         this.numFeatures = parameters.numFeatures;
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
                   ", numFeatures=" + numFeatures +
                   ", numLabels=" + numLabels +
                   ", verbose=" + verbose +
                   ", reportInterval=" + reportInterval +
                   '}';
      }
   }


}//END OF LinearModel
