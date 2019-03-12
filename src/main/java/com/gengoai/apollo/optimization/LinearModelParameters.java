package com.gengoai.apollo.optimization;

import com.gengoai.apollo.linear.p2.NDArray;
import com.gengoai.apollo.optimization.activation.Activation;

/**
 * @author David B. Bracewell
 */
public interface LinearModelParameters {

   default NDArray activate(NDArray input) {
      return getActivation().apply(getWeights()
                                      .mmul(input)
                                      .addiColumnVector(getBias()));
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
