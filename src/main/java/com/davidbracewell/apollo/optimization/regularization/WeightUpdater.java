package com.davidbracewell.apollo.optimization.regularization;

import com.davidbracewell.apollo.optimization.Weights;

/**
 * @author David B. Bracewell
 */
public interface WeightUpdater {

   double update(Weights weights, Weights gradient, double learningRate);

}//END OF WeightUpdater
