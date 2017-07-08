package com.davidbracewell.apollo.optimization;

import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.optimization.update.WeightUpdate;
import com.davidbracewell.function.SerializableSupplier;
import com.davidbracewell.guava.common.util.concurrent.AtomicDouble;
import com.davidbracewell.logging.Loggable;
import com.davidbracewell.stream.MStream;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

import java.io.Serializable;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * @author David B. Bracewell
 */
@NoArgsConstructor
@AllArgsConstructor
public class SGD implements Optimizer, Serializable, Loggable {
   private static final long serialVersionUID = 1L;
   @Getter
   @Setter
   private int reportInterval = 100;

   @Override
   public CostWeightTuple optimize(WeightComponent theta,
                                   SerializableSupplier<MStream<? extends Vector>> stream,
                                   CostFunction costFunction,
                                   TerminationCriteria terminationCriteria,
                                   LearningRate learningRate,
                                   WeightUpdate weightUpdater,
                                   boolean verbose
                                  ) {
      final AtomicDouble lr = new AtomicDouble(learningRate.getInitialRate());
      double lastLoss = 0;
      final AtomicInteger numProcessed = new AtomicInteger(0);
      int iteration = 0;
      for (; iteration < terminationCriteria.maxIterations(); iteration++) {
         final int time = iteration;
         double sumTotal = stream.get().shuffle().mapToDouble(v -> {
            double cost = step(v, theta, costFunction, weightUpdater, lr.get());
            numProcessed.incrementAndGet();
            lr.set(learningRate.get(lr.get(), time, numProcessed.get()));
            return cost;
         }).sum();
         if (verbose && iteration % reportInterval == 0) {
            logInfo("iteration={0}, total_cost={1}", iteration, sumTotal);
         }
         lastLoss = sumTotal;
         if (terminationCriteria.check(sumTotal)) {
            break;
         }
      }
      if (verbose) {
         logInfo("Finished: iteration={0}, total_cost={1}", iteration, terminationCriteria.lastLoss());
      }
      return CostWeightTuple.of(lastLoss, theta);
   }

   private double step(Vector next,
                       WeightComponent theta,
                       CostFunction costFunction,
                       WeightUpdate updater,
                       double lr
                      ) {
      return updater.update(theta, costFunction.evaluate(next, theta), lr);
   }
}//END OF SGD
