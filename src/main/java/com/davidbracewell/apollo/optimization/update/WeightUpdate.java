package com.davidbracewell.apollo.optimization.update;

import com.davidbracewell.apollo.optimization.Gradient;
import com.davidbracewell.apollo.optimization.Weights;

/**
 * @author David B. Bracewell
 */
public interface WeightUpdate {

   double update(Weights weights, Gradient gradient, double learningRate);

}//END OF WeightUpdate
