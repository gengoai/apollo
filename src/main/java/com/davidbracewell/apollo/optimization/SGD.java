package com.davidbracewell.apollo.optimization;

import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.optimization.update.WeightUpdate;
import com.davidbracewell.function.SerializableSupplier;
import com.davidbracewell.guava.common.base.Stopwatch;
import com.davidbracewell.logging.Loggable;
import com.davidbracewell.stream.MStream;
import lombok.val;

import java.io.Serializable;

/**
 * @author David B. Bracewell
 */
public class SGD implements Optimizer, Serializable, Loggable {
   private static final long serialVersionUID = 1L;

   @Override
   public CostWeightTuple optimize(WeightMatrix theta,
                                   SerializableSupplier<MStream<Vector>> stream,
                                   CostFunction costFunction,
                                   TerminationCriteria terminationCriteria,
                                   LearningRate learningRate,
                                   WeightUpdate weightUpdater,
                                   int reportInterval
                                  ) {
      int numProcessed = 0;
      double lr = learningRate.getInitialRate();
      double lastLoss = 0d;
      for (int iteration = 0; iteration < terminationCriteria.maxIterations(); iteration++) {
         val timer = Stopwatch.createStarted();
         double totalLoss = 0;
         for (Vector datum : stream.get()) {
            CostGradientTuple cgt = costFunction.evaluate(datum, theta);
            totalLoss += cgt.getCost() + weightUpdater.update(theta, cgt.getGradient(), lr, iteration);
            numProcessed++;
            lr = learningRate.get(lr, iteration, numProcessed);
         }
         if (reportInterval > 0 && (iteration == 0 || (iteration + 1) % reportInterval == 0 || iteration == terminationCriteria
                                                                                                               .maxIterations() - 1)) {
            logInfo("iteration={0}, loss={1}, time={2}", (iteration + 1), totalLoss, timer);
         }
         lastLoss = totalLoss;
         if ((iteration + 1) < terminationCriteria.maxIterations() && terminationCriteria.check(totalLoss)) {
            break;
         }
      }
      return CostWeightTuple.of(lastLoss, theta);
   }
}// END OF Optimizer
