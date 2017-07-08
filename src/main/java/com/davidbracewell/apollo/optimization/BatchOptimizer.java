package com.davidbracewell.apollo.optimization;

import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.optimization.update.WeightUpdate;
import com.davidbracewell.function.SerializableSupplier;
import com.davidbracewell.guava.common.util.concurrent.AtomicDouble;
import com.davidbracewell.logging.Loggable;
import com.davidbracewell.stream.MStream;
import com.davidbracewell.stream.StreamingContext;
import lombok.*;
import lombok.experimental.Accessors;

import java.io.Serializable;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * @author David B. Bracewell
 */
@Accessors(fluent = true)
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class BatchOptimizer implements Optimizer, Loggable, Serializable {
   private static final long serialVersionUID = 1L;
   @Getter
   @Setter
   @Builder.Default
   private Optimizer subOptimizer = new SGD();
   @Getter
   @Setter
   @Builder.Default
   private int batchSize = 50;
   @Getter
   @Setter
   @Builder.Default
   private int reportInterval = 100;

   @Override
   public CostWeightTuple optimize(WeightComponent theta, SerializableSupplier<MStream<? extends Vector>> stream, CostFunction costFunction, TerminationCriteria terminationCriteria, LearningRate learningRate, WeightUpdate weightUpdater, boolean verbose) {
      int iterations = terminationCriteria.maxIterations();
      terminationCriteria.maxIterations(1);
      double lastLoss = 0;
      AtomicDouble lr = new AtomicDouble(learningRate.getInitialRate());
      AtomicInteger numProcessed = new AtomicInteger(0);
      for (int i = 0; i < iterations; i++) {
         final int iteration = i;
         lastLoss = stream.get().shuffle().split(batchSize).javaStream().sequential().mapToDouble(batch -> {
            final SubUpdate subUpdate = new SubUpdate();
            subOptimizer.optimize(theta, () -> StreamingContext.local().stream(batch),
                                  costFunction, terminationCriteria,
                                  learningRate, subUpdate, false);
            numProcessed.addAndGet(batchSize);
            lr.set(learningRate.get(lr.get(), iteration, numProcessed.get()));
            if (subUpdate.gradient != null) {
               for (Gradient gradient : subUpdate.gradient.getGradients()) {
                  gradient.mapDivideSelf(batchSize);
               }
               weightUpdater.update(theta, subUpdate.gradient, lr.get());
               return subUpdate.totalLoss;
            }
            return 0d;
         }).sum();
         if (verbose && i % reportInterval == 0) {
            logInfo("iteration=" + (i + 1) + ", totalCost=" + lastLoss);
         }
         if (terminationCriteria.check(lastLoss)) {
            break;
         }
      }
      terminationCriteria.maxIterations(iterations);
      return CostWeightTuple.of(lastLoss, theta);
   }

   private static class SubUpdate implements WeightUpdate, Serializable {
      private static final long serialVersionUID = 1L;
      private CostGradientTuple gradient;
      private double totalLoss = 0;

      @Override
      public double update(WeightComponent theta, CostGradientTuple observation, double learningRate) {
         if (gradient == null) {
            this.gradient = observation;
            this.totalLoss = observation.getLoss();
         } else {
            for (int i = 0; i < this.gradient.getGradients().length; i++) {
               this.gradient.getGradient(i).addSelf(observation.getGradient(i));
            }
            this.totalLoss += observation.getLoss();
         }
         return 0;
      }

      @Override
      public double update(Weights weights, Gradient gradient, double learningRate) {
         throw new UnsupportedOperationException();
      }
   }
}//END OF BatchOptimizer
