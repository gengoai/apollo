package com.davidbracewell.apollo.optimization;

import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.optimization.update.WeightUpdate;
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
   public CostWeightTuple optimize(WeightComponent theta, SerializableSupplier<MStream<? extends Vector>> stream, CostFunction costFunction, TerminationCriteria terminationCriteria, LearningRate learningRate, WeightUpdate weightUpdater, boolean verbose) {
      int iterations = terminationCriteria.maxIterations();
      terminationCriteria.maxIterations(1);
      double lastLoss = 0;
      AtomicDouble lr = new AtomicDouble(learningRate.getInitialRate());
      AtomicInteger numProcessed = new AtomicInteger(0);
      for (int i = 0; i < iterations; i++) {
         final int iteration = i;
         lastLoss = stream.get().shuffle().split(batchSize).mapToDouble(batch -> {
            final SubUpdate subUpdate = new SubUpdate();
            CostWeightTuple lwt = subOptimizer.optimize(theta, () -> StreamingContext.local().stream(batch),
                                                        costFunction, terminationCriteria,
                                                        learningRate, subUpdate, false);
            numProcessed.addAndGet(batchSize);
            lr.set(learningRate.get(lr.get(), iteration, numProcessed.get()));
            double totalLoss = lwt.getLoss();
            subUpdate.gradient.getTheta().scaleSelf(1d / batchSize);
            subUpdate.gradient.getBias().mapDivideSelf(batchSize);
            totalLoss += weightUpdater.update(theta.get(0), subUpdate.gradient, lr.get());
            return totalLoss;
         }).sum();


         if (verbose && i % 10 == 0) {
            System.err.println("iteration=" + (i + 1) + ", totalCost=" + lastLoss);
         }
      }
      terminationCriteria.maxIterations(iterations);
      return CostWeightTuple.of(lastLoss, theta);
   }

   private static class SubUpdate implements WeightUpdate, Serializable {
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
}//END OF BatchOptimizer
