package com.davidbracewell.apollo.ml.optimization;

/**
 * @author David B. Bracewell
 */

import com.davidbracewell.apollo.linear.NDArray;

/**
 * @author David B. Bracewell
 */
public interface CostFunction<THETA> {

   CostGradientTuple evaluate(NDArray input, THETA theta);

}//END OF CostFunction