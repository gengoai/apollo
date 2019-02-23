package com.gengoai.apollo.optimization;

import com.gengoai.Stopwatch;
import com.gengoai.apollo.linear.NDArray;
import com.gengoai.function.SerializableSupplier;
import com.gengoai.stream.MStream;

import java.util.Iterator;

/**
 * @author David B. Bracewell
 */
public class GradientDescentOptimizer implements Optimizer<LinearModelParameters> {
   double cost;
   int batchSize;

   @java.beans.ConstructorProperties({"cost", "batchSize"})
   GradientDescentOptimizer(double cost, int batchSize) {
      this.cost = cost;
      this.batchSize = batchSize;
   }

   public static GradientDescentOptimizerBuilder builder() {
      return new GradientDescentOptimizerBuilder();
   }

   public int getBatchSize() {
      return this.batchSize;
   }

   @Override
   public double getFinalCost() {
      return cost;
   }

   @Override
   public void optimize(LinearModelParameters startingTheta,
                        SerializableSupplier<MStream<NDArray>> stream,
                        CostFunction<LinearModelParameters> costFunction,
                        StoppingCriteria stoppingCriteria,
                        WeightUpdate weightUpdater,
                        int reportInterval
                       ) {
      BatchIterator iterator = new BatchIterator(stream.get().collect(),
                                                 startingTheta.numberOfLabels(),
                                                 startingTheta.numberOfFeatures());
      for (int iteration = 0; iteration < stoppingCriteria.maxIterations(); iteration++) {
         cost = 0;
         iterator.shuffle();
         Stopwatch timer = Stopwatch.createStarted();
         for (Iterator<NDArray> batch = iterator.iterator(batchSize); batch.hasNext(); ) {
            NDArray input = batch.next();
            CostGradientTuple cgt = costFunction.evaluate(input, startingTheta);
            cost += cgt.getCost() + weightUpdater.update(startingTheta, cgt.getGradient(), iteration);
         }
         cost /= iterator.size();
         timer.stop();
         if (report(reportInterval, iteration, stoppingCriteria, cost, timer.toString())) {
            break;
         }
      }
   }

   @Override
   public void reset() {
      cost = Double.POSITIVE_INFINITY;
   }

   public void setBatchSize(int batchSize) {
      this.batchSize = batchSize;
   }

   public static class GradientDescentOptimizerBuilder {
      private int batchSize;
      private double cost;

      GradientDescentOptimizerBuilder() {
      }

      public GradientDescentOptimizer.GradientDescentOptimizerBuilder batchSize(int batchSize) {
         this.batchSize = batchSize;
         return this;
      }

      public GradientDescentOptimizer build() {
         return new GradientDescentOptimizer(cost, batchSize);
      }

      public GradientDescentOptimizer.GradientDescentOptimizerBuilder cost(double cost) {
         this.cost = cost;
         return this;
      }

      public String toString() {
         return "GradientDescentOptimizer.GradientDescentOptimizerBuilder(cost=" + this.cost + ", batchSize=" + this.batchSize + ")";
      }
   }
}// END OF SGD
