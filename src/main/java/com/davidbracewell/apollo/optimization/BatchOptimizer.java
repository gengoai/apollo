package com.davidbracewell.apollo.optimization;

import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.optimization.update.WeightUpdate;
import com.davidbracewell.function.SerializableSupplier;
import com.davidbracewell.guava.common.util.concurrent.AtomicDouble;
import com.davidbracewell.logging.Loggable;
import com.davidbracewell.stream.MStream;
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

   @Override
   public CostWeightTuple optimize(WeightMatrix theta,
                                   SerializableSupplier<MStream<Vector>> stream,
                                   CostFunction costFunction,
                                   TerminationCriteria terminationCriteria,
                                   LearningRate learningRate,
                                   WeightUpdate weightUpdater,
                                   int reportInterval
                                  ) {
      int iterations = terminationCriteria.maxIterations();
      terminationCriteria.maxIterations(1);
      double lastLoss = 0;
      AtomicDouble lr = new AtomicDouble(learningRate.getInitialRate());
      AtomicInteger numProcessed = new AtomicInteger(0);
      for (int i = 0; i < iterations; i++) {
         final int iteration = i;
         lastLoss = stream.get().shuffle().split(batchSize).mapToDouble(batch -> {
            final SubUpdate subUpdate = new SubUpdate();

            CostWeightTuple cwt = subOptimizer.optimize(theta, stream,
                                                        costFunction, terminationCriteria,
                                                        learningRate, subUpdate, 0);
            numProcessed.addAndGet(batchSize);
            lr.set(learningRate.get(lr.get(), iteration, numProcessed.get()));


            if (subUpdate.gradient != null) {
               subUpdate.gradient.scale(1d / batchSize);
               return cwt.getCost() + weightUpdater.update(theta, subUpdate.gradient, lr.get());
            }
            return cwt.getCost();
         }).sum();
         if (reportInterval > 0 && i % reportInterval == 0) {
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
      private GradientMatrix gradient;


      @Override
      public double update(WeightMatrix weights, GradientMatrix g, double learningRate) {
         if (gradient == null) {
            this.gradient = g;
         } else {
            this.gradient.add(g);
         }
         return 0;
      }
   }
}//END OF BatchOptimizer
