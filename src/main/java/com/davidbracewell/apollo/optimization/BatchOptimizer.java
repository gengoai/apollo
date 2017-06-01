package com.davidbracewell.apollo.optimization;

import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.optimization.regularization.WeightUpdater;
import com.davidbracewell.stream.MStream;
import com.davidbracewell.stream.StreamingContext;

/**
 * @author David B. Bracewell
 */
public class BatchOptimizer implements Optimizer {
   private final Optimizer subOptimizer;
   private final int batchSize;

   public BatchOptimizer(Optimizer subOptimizer, int batchSize) {
      this.subOptimizer = subOptimizer;
      this.batchSize = batchSize;
   }

   @Override
   public Weights optimize(Weights start, MStream<? extends Vector> stream, StochasticCostFunction costFunction, TerminationCriteria terminationCriteria, LearningRate learningRate, WeightUpdater weightUpdater, boolean verbose) {
      final Weights theta = start.copy();
      int iterations = terminationCriteria.maxIterations();
      terminationCriteria.maxIterations(1);
      for (int i = 0; i < iterations; i++) {
         stream.shuffle().split(batchSize).forEach(batch -> {
            theta.set(subOptimizer.optimize(theta, StreamingContext.local().stream(batch),
                                            costFunction, terminationCriteria,
                                            learningRate, weightUpdater, false));
         });
         if (verbose && i % 10 == 0) {
            System.err.println("iteration=" + (i + 1) + ", totalCost=" + theta.getCost());
         }
      }
      return theta;
   }
}// END OF BatchOptimizer
