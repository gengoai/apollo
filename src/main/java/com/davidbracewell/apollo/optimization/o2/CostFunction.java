package com.davidbracewell.apollo.optimization.o2;

import com.davidbracewell.apollo.optimization.CostGradientTuple;

/**
 * @author David B. Bracewell
 */
public interface CostFunction {

   CostGradientTuple evaluate(WeightComponent theta);

}//END OF CostFunction
