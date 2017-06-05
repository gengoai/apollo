package com.davidbracewell.apollo.optimization.regularization;

import com.davidbracewell.apollo.optimization.Weights;

/**
 * @author David B. Bracewell
 */
public interface Regularizer {

   double update(Weights weights, Weights gradient, double learningRate);

}//END OF Regularizer
