package com.davidbracewell.apollo.optimization;

import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.optimization.update.WeightUpdate;
import com.davidbracewell.function.SerializableSupplier;
import com.davidbracewell.guava.common.base.Stopwatch;
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
         Stopwatch sw = Stopwatch.createStarted();
         final int iteration = i;
         double loss = 0d;
         for (Iterable<Vector> batch : stream.get().split(batchSize)) {
            final SubUpdate subUpdate = new SubUpdate();
            final LearningRate subLearningRate = new ConstantLearningRate(lr.get());
            CostWeightTuple cwt = subOptimizer.optimize(theta, () -> StreamingContext.local().stream(batch),
                                                        costFunction, terminationCriteria,
                                                        subLearningRate, subUpdate, 0);
            numProcessed.addAndGet(batchSize);
            lr.set(learningRate.get(lr.get(), iteration, numProcessed.get()));
            if (subUpdate.gradient != null) {
               subUpdate.gradient.scale(1d / batchSize);
               synchronized (this) {
                  loss += cwt.getCost() / batchSize +
                             weightUpdater.update(theta, subUpdate.gradient, lr.get(), iteration) / batchSize;
               }
            } else {
               loss += cwt.getCost();
            }
         }
         sw.stop();
         if (reportInterval > 0 && ((i + 1) == terminationCriteria.maxIterations() || (i + 1) % reportInterval == 0)) {
            logInfo("iteration={0}, totalLoss={1}, time={2}", (i + 1), loss, sw);
         }
         lastLoss = loss;
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
      public double update(WeightMatrix weights, GradientMatrix g, double learningRate, int iteration) {
         if (gradient == null) {
            this.gradient = g;
         } else {
            this.gradient.add(g);
         }
         return 0;
      }
   }
}//END OF BatchOptimizer
