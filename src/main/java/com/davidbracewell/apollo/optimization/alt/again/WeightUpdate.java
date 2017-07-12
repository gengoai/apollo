package com.davidbracewell.apollo.optimization.alt.again;


/**
 * @author David B. Bracewell
 */
public interface WeightUpdate {

   double update(WeightMatrix weights, GradientMatrix gradient, double learningRate);

}//END OF WeightUpdate
