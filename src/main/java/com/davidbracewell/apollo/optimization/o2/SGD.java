package com.davidbracewell.apollo.optimization.o2;

import com.davidbracewell.apollo.linalg.Matrix;
import com.davidbracewell.apollo.linalg.SparseMatrix;
import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.optimization.CostGradientTuple;
import com.davidbracewell.apollo.optimization.LearningRate;
import com.davidbracewell.apollo.optimization.TerminationCriteria;
import com.davidbracewell.apollo.optimization.Weights;
import com.davidbracewell.apollo.optimization.update.WeightUpdate;
import com.davidbracewell.function.SerializableSupplier;
import com.davidbracewell.guava.common.util.concurrent.AtomicDouble;
import com.davidbracewell.logging.Loggable;
import com.davidbracewell.stream.MStream;

import java.io.Serializable;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * @author David B. Bracewell
 */
public class SGD implements Optimizer, Serializable, Loggable {
   private static final long serialVersionUID = 1L;

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
         double sumTotal = stream.get().mapToDouble(v -> {
            double cost = step(v, theta, costFunction, weightUpdater, lr.get());
            numProcessed.incrementAndGet();
            lr.set(learningRate.get(lr.get(), time, numProcessed.get()));
            return cost;
         }).sum();
         if (verbose && iteration % 10 == 0) {
            logInfo("iteration={0}, total_cost={1}", iteration, sumTotal);
         }
         lastLoss = sumTotal;
         if (terminationCriteria.check(sumTotal)) {
            break;
         }
      }
      if (verbose) {
         logInfo("Finished: totalIterations={0}, total_cost={1}", iteration, terminationCriteria.lastLoss());
      }
      return CostWeightTuple.of(lastLoss, theta);
   }

   private double step(Vector next,
                       WeightComponent theta,
                       CostFunction costFunction,
                       WeightUpdate updater,
                       double lr
                      ) {
      CostGradientTuple observation = costFunction.evaluate(next, theta);
      Vector nextEta = next.mapMultiply(lr);
      Matrix m = observation.getGradient()
                            .toDiagMatrix()
                            .multiply(new SparseMatrix(observation.getGradient().dimension(), nextEta));
      double regLoss = 0;
      for (Weights weights : theta) {
         regLoss += updater.update(weights, new Weights(m, observation.getGradient(), weights.isBinary()), lr);
      }
      return observation.getLoss() + regLoss;
   }
}//END OF SGD
