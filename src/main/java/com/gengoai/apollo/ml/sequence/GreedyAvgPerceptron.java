package com.gengoai.apollo.ml.sequence;

import com.gengoai.apollo.linear.Axis;
import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.linear.NDArrayFactory;
import com.gengoai.apollo.ml.FitParameters;
import com.gengoai.apollo.optimization.TerminationCriteria;
import com.gengoai.collection.Iterables;
import com.gengoai.conversion.Cast;
import com.gengoai.function.SerializableSupplier;
import com.gengoai.stream.MStream;

/**
 * The type Greedy avg perceptron.
 *
 * @author David B. Bracewell
 */
public class GreedyAvgPerceptron implements SequenceLabeler {
   private NDArray bias;
   private NDArray featureWeights;
   private NDArray transitionWeights;

   @Override
   public NDArray estimate(NDArray data) {
      NDArray out = NDArrayFactory.DEFAULT().zeros(data.numRows(), 1, data.numKernels(), data.numChannels());
      data.sliceStream().forEach(t -> {
         int index = t.v1;
         int pLabel = (int) bias.length();
         for (int row = 0; row < t.v2.numRows(); row++) {
            pLabel = (int) predict(t.v2.getVector(row, Axis.ROW), pLabel);
            out.getSlice(index).set(row, pLabel);
         }
      });
      data.setPredicted(out);
      return data;
   }

   @Override
   public void fit(SerializableSupplier<MStream<NDArray>> dataSupplier, FitParameters parameters) {
      fit(dataSupplier, Cast.as(parameters, Parameters.class));

   }

   /**
    * Fit.
    *
    * @param dataSupplier the data supplier
    * @param parameters   the parameters
    */
   public void fit(SerializableSupplier<MStream<NDArray>> dataSupplier, Parameters parameters) {
      this.featureWeights = NDArrayFactory.SPARSE.zeros(parameters.numFeatures, parameters.numLabels);
      this.transitionWeights = NDArrayFactory.SPARSE.zeros(parameters.numLabels + 1, parameters.numLabels);
      this.bias = NDArrayFactory.SPARSE.zeros(parameters.numLabels);

      final NDArray fTotals = featureWeights.copy();
      final NDArray tTotals = transitionWeights.copy();
      final NDArray bTotals = bias.copy();
      final NDArray fTimestamps = featureWeights.copy();
      final NDArray tTimestamps = transitionWeights.copy();
      final NDArray bTimestamps = bias.copy();

      int instances = 0;
      TerminationCriteria terminationCriteria = TerminationCriteria.create()
                                                                   .historySize(parameters.historySize)
                                                                   .maxIterations(parameters.maxIterations)
                                                                   .tolerance(parameters.eps);
      for (int i = 0; i < terminationCriteria.maxIterations(); i++) {
         double total = 0;
         double correct = 0;

         for (NDArray sequence : dataSupplier.get().shuffle()) {
            int pLabel = parameters.numLabels;
            NDArray y = sequence.getLabelAsNDArray().argMax(Axis.ROW);
            for (int row = 0; row < sequence.numRows(); row++) {
               final int predicted = (int) predict(sequence.getVector(row, Axis.ROW), pLabel);
               final int gold = (int) y.get(row);
               total++;
               if (predicted != gold) {
                  instances++;
                  for (NDArray.Entry e : Iterables.asIterable(sequence.sparseRowIterator(row))) {
                     update(gold, e.getColumn(), 1.0, instances, featureWeights, fTimestamps, fTotals);
                     update(predicted, e.getColumn(), -1.0, instances, featureWeights, fTimestamps, fTotals);
                  }
                  update(gold, pLabel, 1.0, instances, transitionWeights, tTimestamps, tTotals);
                  update(predicted, pLabel, -1.0, instances, transitionWeights, tTimestamps, tTotals);
                  updateBias(gold, 1.0, instances, bTimestamps, bTotals);
                  updateBias(predicted, -1.0, instances, bTimestamps, bTotals);
               } else {
                  correct++;
               }
               pLabel = gold;
            }
         }
         double acc = (correct / total) * 100;
         System.out.println(String.format("Iteration %d complete. %.3f accuracy", i + 1, acc));
         if (terminationCriteria.check(100 - acc)) {
            break;
         }
      }

      fTotals.addi(fTimestamps.rsub(instances).mul(featureWeights));
      tTotals.addi(tTimestamps.rsub(instances).mul(transitionWeights));
      bTotals.addi(bTimestamps.rsub(instances).mul(bias));
      featureWeights = fTotals.div(instances);
      transitionWeights = tTotals.div(instances);
      bias = bTotals.div(instances);

   }

   @Override
   public Parameters getDefaultFitParameters() {
      return new Parameters();
   }

   @Override
   public int getNumberOfFeatures() {
      return featureWeights.numRows();
   }

   @Override
   public int getNumberOfLabels() {
      return (int) bias.length();
   }


   /**
    * Custom fit parameters for the GreedyAveragePerceptron
    */
   public static class Parameters extends FitParameters {
      private static final long serialVersionUID = 1L;
      /**
       * The maximum number of iterations to run for
       */
      public int maxIterations = 100;
      /**
       * The epsilon to use for checking for convergence.
       */
      public double eps = 1e-5;

      /**
       * The number of iterations to use for determining convergence
       */
      public int historySize = 3;
   }


   private float predict(NDArray row, int pLabel) {
      NDArray scores = row.mmul(featureWeights);
      scores.addi(transitionWeights.getVector(pLabel, Axis.ROW))
            .addi(bias);
      return scores.argMax(Axis.ROW).get(0);
   }

   private void update(int cls, int feature, double value, int iteration, NDArray weights, NDArray timeStamp, NDArray totals) {
      int iterAt = iteration - (int) timeStamp.get(feature, cls);
      totals.increment(feature, cls, iterAt * weights.get(feature, cls));
      weights.increment(feature, cls, value);
      timeStamp.set(feature, cls, iteration);
   }

   private void updateBias(int cls, double value, int iteration, NDArray timeStamp, NDArray totals) {
      int iterAt = iteration - (int) timeStamp.get(cls);
      totals.increment(cls, iterAt * bias.get(cls));
      bias.increment(cls, value);
      timeStamp.set(cls, iteration);
   }
}//END OF WindowSequenceLabeler
