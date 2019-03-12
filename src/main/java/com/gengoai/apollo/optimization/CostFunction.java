package com.gengoai.apollo.optimization;

import com.gengoai.apollo.linear.p2.NDArray;

/**
 * @author David B. Bracewell
 */
public interface CostFunction<THETA> {

   CostGradientTuple evaluate(NDArray input, THETA theta);

}//END OF CostFunction