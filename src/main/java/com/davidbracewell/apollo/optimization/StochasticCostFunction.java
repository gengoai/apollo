package com.davidbracewell.apollo.optimization;

import com.davidbracewell.apollo.linalg.Vector;

/**
 * @author David B. Bracewell
 */
public interface StochasticCostFunction {

   CostGradientTuple observe(Vector next, Vector weights);

}// END OF StochasticCostFunction
