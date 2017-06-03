package com.davidbracewell.apollo.optimization;

import com.davidbracewell.apollo.linalg.Matrix;
import com.davidbracewell.apollo.linalg.SparseMatrix;
import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.optimization.regularization.WeightUpdater;
import com.davidbracewell.function.SerializableSupplier;
import com.davidbracewell.logging.Logger;
import com.davidbracewell.stream.MStream;
import lombok.Getter;
import lombok.Setter;

/**
 * @author David B. Bracewell
 */
public class SGD implements Optimizer {
   private static final Logger LOG = Logger.getLogger(SGD.class);
   @Getter
   @Setter
   private double alpha = 1;
   @Getter
   @Setter
   private double tolerance = 1e-10;

   private Weights weights;

   @Override
   public LossWeightTuple optimize(Weights start,
                                   SerializableSupplier<MStream<? extends Vector>> stream,
                                   StochasticCostFunction costFunction,
                                   TerminationCriteria terminationCriteria,
                                   LearningRate learningRate,
                                   WeightUpdater weightUpdater,
                                   boolean verbose
                                  ) {
      weights = start.copy();
      int pass = 0;
      int lastPass = 0;
      int numProcessed = 1;
      double lr = learningRate.getInitialRate();
      double lastLoss = 0;
      for (; pass < terminationCriteria.maxIterations(); pass++) {
         final double eta = learningRate.get(lr, 0, numProcessed);
         lr = eta;
         double sumTotal = stream.get()
                                 .shuffle()
                                 .mapToDouble(next -> {
                                    synchronized (this) {
                                       return step(next, costFunction, weightUpdater, verbose, eta);
                                    }
                                 })
                                 .sum();
         lastLoss = sumTotal;
         if (verbose && pass % 10 == 0) {
            LOG.info("pass={0}, total_cost={1}", pass, sumTotal);
         }
         if (terminationCriteria.check(sumTotal)) {
            break;
         }
         lastPass = pass;
      }
      if (verbose && pass != lastPass) {
         LOG.info("pass={0}, total_cost={1}", pass, terminationCriteria.lastLoss());
      }
      return LossWeightTuple.of(lastLoss, weights);
   }

   private double step(Vector next, StochasticCostFunction costFunction, WeightUpdater updater, boolean verbose, double lr) {
      LossGradientTuple observation = costFunction.observe(next, weights);
      Vector nextEta = next.mapMultiply(lr);
      Matrix m = observation.getGradient()
                            .toDiagMatrix()
                            .multiply(new SparseMatrix(observation.getGradient().dimension(), nextEta));
      return observation.getLoss() + updater.update(weights,
                                                    new Weights(m, observation.getGradient(), weights.isBinary()),
                                                    lr);
   }


}// END OF SGD
