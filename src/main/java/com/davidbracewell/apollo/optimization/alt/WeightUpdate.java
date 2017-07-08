package com.davidbracewell.apollo.optimization.alt;


/**
 * @author David B. Bracewell
 */
public interface WeightUpdate {

   double update(WeightVector weights, Gradient gradient, double learningRate);




}//END OF WeightUpdate
