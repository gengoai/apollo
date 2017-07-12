package com.davidbracewell.apollo.optimization.update;


import com.davidbracewell.apollo.optimization.GradientMatrix;
import com.davidbracewell.apollo.optimization.WeightMatrix;

/**
 * @author David B. Bracewell
 */
public interface WeightUpdate {

   double update(WeightMatrix weights, GradientMatrix gradient, double learningRate);

}//END OF WeightUpdate
