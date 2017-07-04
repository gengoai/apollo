package com.davidbracewell.apollo.optimization;

import com.davidbracewell.apollo.linalg.SparseMatrix;
import com.davidbracewell.apollo.linalg.Vector;

import java.util.Iterator;

/**
 * @author David B. Bracewell
 */
public interface CostFunction {

   default CostGradientTuple evaluate(Iterable<Vector> vectors, WeightComponent theta) {
      Iterator<Vector> itr = vectors.iterator();
      if (!itr.hasNext()) {
         return CostGradientTuple.of(0,
                                     Gradient.of(SparseMatrix.zeroes(1, theta.get(0).getBias().dimension()),
                                                 Vector.sZeros(theta.get(0).getBias().dimension())));
      }
      int numExamples = 1;
      CostGradientTuple tuple = evaluate(itr.next(), theta);
      double totalLoss = tuple.getLoss();
      Gradient[] gradients = null;
      while (itr.hasNext()) {
         numExamples++;
         CostGradientTuple cgt = evaluate(itr.next(), theta);
         if (gradients == null) {
            gradients = cgt.getGradients();
         } else {
            for (int i = 0; i < gradients.length; i++) {
               gradients[i].addSelf(cgt.getGradient(i));
            }
         }
         totalLoss += cgt.getLoss();
      }
      if (gradients != null) {
         for (Gradient gradient : gradients) {
            gradient.mapDivideSelf(numExamples);
         }
      }
      return CostGradientTuple.of(totalLoss, gradients);
   }

   CostGradientTuple evaluate(Vector vector, WeightComponent theta);

}//END OF CostFunction
