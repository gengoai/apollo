package com.davidbracewell.apollo.optimization;

import com.davidbracewell.apollo.linalg.Vector;

/**
 * @author David B. Bracewell
 */
public interface DifferentiableLossFunction extends LossFunction {

   Vector gradient(Vector features, double p, double y);

}//END OF DifferentiableLossFunction
