package com.davidbracewell.apollo.optimization.alt;

import com.davidbracewell.apollo.linalg.Vector;

/**
 * @author David B. Bracewell
 */
public interface CostFunction {

   CostGradientTuple evaluate(Vector vector, WeightVector theta);

}//END OF CostFunction
