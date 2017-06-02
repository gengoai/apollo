package com.davidbracewell.apollo.optimization;

import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.optimization.regularization.WeightUpdater;
import com.davidbracewell.function.SerializableSupplier;
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
   public LossWeightTuple optimize(Weights start, SerializableSupplier<MStream<? extends Vector>> stream, StochasticCostFunction costFunction, TerminationCriteria terminationCriteria, LearningRate learningRate, WeightUpdater weightUpdater, boolean verbose) {
      final Weights theta = start.copy();
      int iterations = terminationCriteria.maxIterations();
      terminationCriteria.maxIterations(1);
      double lastLoss = 0;
      for (int i = 0; i < iterations; i++) {
         lastLoss = stream.get().shuffle().split(batchSize).mapToDouble(batch -> {
            LossWeightTuple lwt = subOptimizer.optimize(theta, () -> StreamingContext.local().stream(batch),
                                                        costFunction, terminationCriteria,
                                                        learningRate, weightUpdater, false);
            theta.set(lwt.getWeights());
            return lwt.getLoss();
         }).sum();
         if (verbose && i % 10 == 0) {
            System.err.println("iteration=" + (i + 1) + ", totalCost=" + lastLoss);
         }
      }
      return LossWeightTuple.of(lastLoss, theta);
   }
}// END OF BatchOptimizer
