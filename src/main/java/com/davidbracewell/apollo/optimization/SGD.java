package com.davidbracewell.apollo.optimization;

import com.davidbracewell.apollo.analysis.Optimum;
import com.davidbracewell.apollo.linalg.Matrix;
import com.davidbracewell.apollo.linalg.SparseMatrix;
import com.davidbracewell.apollo.linalg.SparseVector;
import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.optimization.activation.Activation;
import com.davidbracewell.logging.Logger;
import com.davidbracewell.stream.MStream;
import com.davidbracewell.stream.StreamingContext;
import lombok.Data;
import lombok.Getter;
import lombok.Setter;

/**
 * @author David B. Bracewell
 */
public class SGD implements Minimizer {
   private static final Logger LOG = Logger.getLogger(SGD.class);
   @Getter
   @Setter
   private double alpha = 1;
   @Getter
   @Setter
   private double tolerance = 1e-10;

   private Weights weights;
   private GradientDescentUpdater updater = new GradientDescentUpdater();

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

      Weights weights = sgd.minimize(SparseVector.zeros(10),
                                     StreamingContext.local().stream(vectors),
                                     new LogisticCostFunction(),
                                     100,
                                     true);

      final Activation activation = new LogisticCostFunction().activation;
      System.out.println(weights.getTheta());
      for (Vector v : vectors) {
         Vector p = activation.apply(weights.dot(v));
         System.out.println(Optimum.MAXIMUM.optimumIndex(p.toArray()) + " : " + v.getLabel());
      }


   }

   @Override
   public Weights minimize(Vector start, MStream<Vector> stream, StochasticCostFunction costFunction, int numPasses, boolean verbose) {
      weights = Weights.multiClass(3, start.dimension());
      double l1 = Double.MAX_VALUE;
      double l2 = Double.MAX_VALUE;
      final OptInfo optInfo = new OptInfo(0, alpha);
      int pass = 0;
      int lastPass = 0;
      for (; pass < numPasses; pass++) {
         double sumTotal = stream.shuffle()
                                 .mapToDouble(next -> step(next, costFunction, verbose, optInfo))
                                 .sum();
         if (verbose && pass % 10 == 0) {
            LOG.info("pass={0}, total_cost={1}", pass, sumTotal);
         }
         if (Math.abs(sumTotal - l1) < tolerance && Math.abs(sumTotal - l2) < tolerance) {
            l1 = sumTotal;
            break;
         }
         l2 = l1;
         l1 = sumTotal;
         lastPass = pass;
         optInfo.et0 *= 0.95;
      }
      if (verbose && pass != lastPass) {
         LOG.info("pass={0}, total_cost={1}", pass, l1);
      }
      return weights;
   }

   private double step(Vector next, StochasticCostFunction costFunction, boolean verbose, OptInfo optinfo) {
      LossGradientTuple observation = costFunction.observe(next, weights);
      Vector nextEta = next.mapMultiply(optinfo.et0);
      Matrix m = observation.getGradient()
                            .toDiagMatrix()
                            .multiply(new SparseMatrix(observation.getGradient().dimension(), nextEta));
//      updater.update(weights, new Weights(m, observation.getGradient(), weights.isBinary()));
      return observation.getLoss();
   }

   @Data
   private static class OptInfo {
      long numProcessed;
      double et0;

      public OptInfo(long numProcessed, double et0) {
         this.numProcessed = numProcessed;
         this.et0 = et0;
      }
   }


}// END OF SGD
