package com.gengoai.apollo.optimization;

import com.gengoai.apollo.linear.NDArray;

/**
 * @author David B. Bracewell
 */
public interface CostFunction<THETA> {

   CostGradientTuple evaluate(NDArray input, THETA theta);

}//END OF CostFunction