package com.davidbracewell.apollo.ml.classification.nn;

import com.davidbracewell.apollo.linear.NDArray;
import com.davidbracewell.apollo.ml.optimization.CostFunction;
import com.davidbracewell.apollo.ml.optimization.CostGradientTuple;

/**
 * @author David B. Bracewell
 */
public class FeedForwardCostFunction implements CostFunction<FeedForwardNetwork> {
   @Override
   public CostGradientTuple evaluate(NDArray input, FeedForwardNetwork network) {
      return null;
   }
}// END OF FeedForwardCostFunction
