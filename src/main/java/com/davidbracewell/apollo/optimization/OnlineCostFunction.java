package com.davidbracewell.apollo.optimization;

import com.davidbracewell.apollo.linalg.Vector;

/**
 * The interface Online cost function.
 *
 * @author David B. Bracewell
 */
public interface OnlineCostFunction {

   /**
    * Observe loss valueGradient tuple.
    *
    * @param next    the next
    * @param weights the weights
    * @return the loss valueGradient tuple
    */
   CostGradientTuple observe(Vector next, Weights weights);

}// END OF OnlineCostFunction
