package com.davidbracewell.apollo.ml.classification.nn;

import com.davidbracewell.apollo.optimization.WeightInitializer;
import com.davidbracewell.apollo.optimization.activation.Activation;
import lombok.NonNull;

/**
 * @author David B. Bracewell
 */
public class DenseLayer extends WeightLayer {
   private static final long serialVersionUID = 1L;


   public DenseLayer(int inputSize, int outputSize, @NonNull Activation activation, @NonNull WeightInitializer weightInitializer) {
      super(inputSize, outputSize, activation, weightInitializer);
   }

   public static Builder relu() {
      return new Builder().activation(Activation.RELU);
   }

   public static Builder sigmoid() {
      return new Builder().activation(Activation.SIGMOID);
   }

   public static class Builder extends WeightLayerBuilder<Builder> {

      @Override
      public Layer build() {
         return new DenseLayer(getInputSize(), getOutputSize(), getActivation(), getWeightInitializer());
      }
   }


}// END OF DenseLayer
