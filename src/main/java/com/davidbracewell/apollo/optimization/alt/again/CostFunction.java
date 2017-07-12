package com.davidbracewell.apollo.optimization.alt.again;

import com.davidbracewell.apollo.linalg.Vector;

/**
 * @author David B. Bracewell
 */
public interface CostFunction {

   CostGradientTuple evaluate(Vector vector, WeightMatrix theta);

}//END OF CostFunction
