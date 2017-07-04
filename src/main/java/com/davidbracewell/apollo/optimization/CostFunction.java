package com.davidbracewell.apollo.optimization;

import com.davidbracewell.apollo.linalg.Vector;

import java.util.Iterator;

/**
 * @author David B. Bracewell
 */
public interface CostFunction {

   default CostGradientTuple evaluate(Iterable<Vector> vectors, WeightComponent theta) {
      Iterator<Vector> itr = vectors.iterator();
      if (!itr.hasNext()) {
         return CostGradientTuple.of(0, Vector.sZeros(theta.get(0).getBias().dimension()));
      }
      int numExamples = 1;
      CostGradientTuple tuple = evaluate(itr.next(), theta);
      double totalLoss = tuple.getLoss();
      while (itr.hasNext()) {
         numExamples++;
         CostGradientTuple cgt = evaluate(itr.next(), theta);
         tuple.getGradient().addSelf(cgt.getGradient());
         totalLoss += cgt.getLoss();
      }
      return CostGradientTuple.of(totalLoss, tuple.getGradient().mapDivideSelf(numExamples));
   }

   CostGradientTuple evaluate(Vector vector, WeightComponent theta);

}//END OF CostFunction
