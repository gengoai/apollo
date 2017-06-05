package com.davidbracewell.apollo.optimization;

import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.optimization.regularization.Regularizer;
import com.davidbracewell.function.SerializableSupplier;
import com.davidbracewell.guava.common.util.concurrent.AtomicDouble;
import com.davidbracewell.stream.MStream;
import com.davidbracewell.stream.StreamingContext;

import java.io.Serializable;
import java.util.concurrent.atomic.AtomicInteger;

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
   public LossWeightTuple optimize(Weights start, SerializableSupplier<MStream<? extends Vector>> stream, StochasticCostFunction costFunction, TerminationCriteria terminationCriteria, LearningRate learningRate, Regularizer weightUpdater, boolean verbose) {
      final Weights theta = start.copy();
      int iterations = terminationCriteria.maxIterations();
      terminationCriteria.maxIterations(1);
      double lastLoss = 0;
      AtomicDouble lr = new AtomicDouble(learningRate.getInitialRate());
      AtomicInteger numProcessed = new AtomicInteger(0);
      for (int i = 0; i < iterations; i++) {
         final int iteration = i;
         lastLoss = stream.get().shuffle().split(batchSize).mapToDouble(batch -> {
            final SubUpdate subUpdate = new SubUpdate();
            LossWeightTuple lwt = subOptimizer.optimize(theta, () -> StreamingContext.local().stream(batch),
                                                        costFunction, terminationCriteria,
                                                        learningRate, subUpdate, false);
//            theta.set(lwt.getWeights());
            lr.set(learningRate.get(lr.get(), iteration, numProcessed.get()));
            if (subUpdate.gradient != null) {
               return lwt.getLoss() + weightUpdater.update(theta, subUpdate.gradient, lr.get());
            }
            return 0d;
         }).sum();


         if (verbose && i % 10 == 0) {
            System.err.println("iteration=" + (i + 1) + ", totalCost=" + lastLoss);
         }
      }
      terminationCriteria.maxIterations(iterations);
      return LossWeightTuple.of(lastLoss, theta);
   }


   private static class SubUpdate implements Regularizer, Serializable {
      private static final long serialVersionUID = 1L;
      private Weights gradient;

      @Override
      public double update(Weights weights, Weights gradient, double learningRate) {
         if (this.gradient == null) {
            this.gradient = gradient;
         } else {
            this.gradient.getTheta().addSelf(gradient.getTheta());
            this.gradient.getBias().addSelf(gradient.getBias());
         }
         return 0;
      }
   }

}// END OF BatchOptimizer
