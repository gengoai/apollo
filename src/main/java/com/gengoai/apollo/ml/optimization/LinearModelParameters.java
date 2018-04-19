package com.gengoai.apollo.ml.optimization;

import com.gengoai.apollo.linear.Axis;
import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.ml.optimization.activation.Activation;

/**
 * @author David B. Bracewell
 */
public interface LinearModelParameters {

   default NDArray activate(NDArray input) {
      return getActivation().apply(getWeights().mmul(input).addi(getBias(), Axis.COlUMN));
   }

   Activation getActivation();

   NDArray getBias();

   default int getNumberOfWeightVectors() {
      return isBinary() ? 1 : numberOfLabels();
   }

   NDArray getWeights();

   default boolean isBinary() {
      return numberOfLabels() <= 2;
   }

   int numberOfFeatures();

   int numberOfLabels();

}//END OF LinearModelParameters
