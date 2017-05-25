package com.davidbracewell.apollo.optimization;

import com.davidbracewell.apollo.linalg.SparseVector;
import com.davidbracewell.apollo.linalg.Vector;
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

   private Vector theta;

   public static void main(String[] args) {
      SGD sgd = new SGD();

      Vector[] vectors = {
         SparseVector.zeros(10).set(0, 1).set(9, 1).withLabel(1.0),
         SparseVector.zeros(10).set(1, 1).set(9, 1).withLabel(1.0),
         SparseVector.zeros(10).set(2, 1).set(9, 1).withLabel(1.0),
         SparseVector.zeros(10).set(3, 1).set(9, 1).withLabel(1.0),
         SparseVector.zeros(10).set(0, 1).set(9, 1).withLabel(1.0),
         SparseVector.zeros(10).set(4, 1).set(9, 1).withLabel(1.0),
         SparseVector.zeros(10).set(5, 1).set(9, 1).withLabel(1.0),
         SparseVector.zeros(10).set(3, 1).set(9, 1).withLabel(1.0),
         SparseVector.zeros(10).set(5, 1).set(9, 1).withLabel(0.0),
         SparseVector.zeros(10).set(6, 1).set(9, 1).withLabel(0.0),
         SparseVector.zeros(10).set(7, 1).set(9, 1).withLabel(0.0),
         SparseVector.zeros(10).set(8, 1).set(9, 1).withLabel(0.0),
         SparseVector.zeros(10).set(5, 1).set(9, 1).withLabel(0.0),
         SparseVector.zeros(10).set(6, 1).set(9, 1).withLabel(0.0),
         SparseVector.zeros(10).set(7, 1).set(9, 1).withLabel(0.0),
         SparseVector.zeros(10).set(8, 1).set(9, 1).withLabel(0.0),

      };

      Vector weights = sgd.minimize(SparseVector.zeros(10),
                                    StreamingContext.local().stream(vectors),
                                    new LogisticCostFunction(),
                                    200,
                                    true);

      System.out.println(weights);
      final Activation activation = new LogisticCostFunction().activation;
      for (Vector v : vectors) {
         double p = weights.dot(v);
         System.out.println(v.getLabel() + " : " + activation.apply(p) + " : " + p);
      }

   }

   @Override
   public Vector minimize(Vector start, MStream<Vector> stream, StochasticCostFunction costFunction, int numPasses, boolean verbose) {
      theta = start.copy();
      double l1 = Double.MAX_VALUE;
      double l2 = Double.MAX_VALUE;
      final OptInfo optInfo = new OptInfo(0, alpha);
      for (int pass = 0; pass < numPasses; pass++) {
         double sumTotal = stream.shuffle()
                                 .mapToDouble(next -> step(next, theta, costFunction, verbose, optInfo))
                                 .sum();
         if (verbose && pass % 10 == 0) {
            LOG.info("pass={0}, total_cost={1}, w_norm={2}", pass, sumTotal, theta.magnitude());
         }
         if (Math.abs(sumTotal - l1) < tolerance && Math.abs(sumTotal - l2) < tolerance) {
            break;
         }
         l2 = l1;
         l1 = sumTotal;
         optInfo.et0 *= 0.95;
      }
      return theta;
   }

   private double step(Vector next, Vector weights, StochasticCostFunction costFunction, boolean verbose, OptInfo optinfo) {
      CostGradientTuple observation = costFunction.observe(next, weights);
      theta = theta.add(observation.getGradient().mapMultiply(optinfo.et0));
      return observation.getCost();
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
