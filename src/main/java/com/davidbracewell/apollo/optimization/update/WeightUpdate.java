package com.davidbracewell.apollo.optimization.update;

import com.davidbracewell.apollo.optimization.CostGradientTuple;
import com.davidbracewell.apollo.optimization.Gradient;
import com.davidbracewell.apollo.optimization.WeightComponent;
import com.davidbracewell.apollo.optimization.Weights;

/**
 * @author David B. Bracewell
 */
public interface WeightUpdate {

   default double update(WeightComponent theta, CostGradientTuple observation, double learningRate) {
      double extraLoss = 0;
      for (int i = 0; i < theta.size(); i++) {
         Weights weights = theta.get(i);
         Gradient gradient = observation.getGradient(i);
         extraLoss += update(weights, gradient, learningRate);
      }
      return extraLoss;
   }

   double update(Weights weights, Gradient gradient, double learningRate);

}//END OF WeightUpdate
