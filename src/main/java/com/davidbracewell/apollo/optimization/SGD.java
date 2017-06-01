package com.davidbracewell.apollo.optimization;

import com.davidbracewell.apollo.linalg.Matrix;
import com.davidbracewell.apollo.linalg.SparseMatrix;
import com.davidbracewell.apollo.linalg.SparseVector;
import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.optimization.activation.Activation;
import com.davidbracewell.apollo.optimization.regularization.L1Regularization;
import com.davidbracewell.apollo.optimization.regularization.WeightUpdater;
import com.davidbracewell.logging.Logger;
import com.davidbracewell.stream.MStream;
import com.davidbracewell.stream.StreamingContext;
import lombok.Data;
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



   public static void main(String[] args) {
      SGD sgd = new SGD();

      Vector[] vectors = {
         SparseVector.zeros(10).set(0, 1).withLabel(1.0),
         SparseVector.zeros(10).set(1, 1).withLabel(1.0),
         SparseVector.zeros(10).set(2, 1).withLabel(1.0),
         SparseVector.zeros(10).set(3, 1).withLabel(1.0),
         SparseVector.zeros(10).set(0, 1).withLabel(1.0),
         SparseVector.zeros(10).set(4, 1).withLabel(1.0),
         SparseVector.zeros(10).set(5, 1).withLabel(1.0),
         SparseVector.zeros(10).set(3, 1).withLabel(1.0),
         SparseVector.zeros(10).set(5, 1).withLabel(0.0),
         SparseVector.zeros(10).set(6, 1).withLabel(0.0),
         SparseVector.zeros(10).set(7, 1).withLabel(0.0),
         SparseVector.zeros(10).set(8, 1).withLabel(0.0),
         SparseVector.zeros(10).set(5, 1).withLabel(0.0),
         SparseVector.zeros(10).set(6, 1).withLabel(0.0),
         SparseVector.zeros(10).set(7, 1).withLabel(0.0),
         SparseVector.zeros(10).set(8, 1).withLabel(0.0),
         SparseVector.zeros(10).set(9, 1).withLabel(2.0),
         SparseVector.zeros(10).set(9, 1).withLabel(2.0),
         SparseVector.zeros(10).set(9, 1).withLabel(2.0),
         SparseVector.zeros(10).set(9, 1).withLabel(2.0),
         SparseVector.zeros(10).set(9, 1).withLabel(2.0),
         SparseVector.zeros(10).set(9, 1).withLabel(2.0),
         SparseVector.zeros(10).set(9, 1).withLabel(2.0),
         SparseVector.zeros(10).set(9, 1).withLabel(2.0),

      };

      LearningRate learningRate = new DecayLearningRate(0.1, 0.01);
      WeightUpdater updater = new L1Regularization(0.01);
      Weights weights = sgd.optimize(Weights.multiClass(3, 10),
                                     StreamingContext.local().stream(vectors),
                                     new LogisticCostFunction(),
                                     TerminationCriteria.create().maxIterations(20),
                                     learningRate,
                                     updater,
                                     true);

      final Activation activation = new LogisticCostFunction().activation;
      for (Vector v : vectors) {
         Vector p = activation.apply(weights.dot(v));
         System.out.println(Optimum.MAXIMUM.optimumIndex(p.toArray()) + " : " + v.getLabel());
      }

   }

   @Override
   public Weights optimize(Weights start,
                           MStream<? extends Vector> stream,
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
      for (; pass < terminationCriteria.maxIterations(); pass++) {
         final double eta = learningRate.get(lr, 0, numProcessed);
         lr = eta;
         double sumTotal = stream.shuffle()
                                 .mapToDouble(next -> step(next, costFunction, weightUpdater, verbose, eta))
                                 .sum();
         weights.setCost(sumTotal);
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
      return weights;
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

   @Data
   private static class OptInfo {
      int numProcessed;
      int iteration;
      double et0;

      public OptInfo(int numProcessed, double et0, int iteration) {
         this.numProcessed = numProcessed;
         this.et0 = et0;
         this.iteration = iteration;
      }
   }


}// END OF SGD
