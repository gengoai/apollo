package com.davidbracewell.apollo.ml.classification.nn;


import com.davidbracewell.apollo.ml.optimization.WeightInitializer;
import com.davidbracewell.apollo.ml.optimization.activation.Activation;

/**
 * @author David B. Bracewell
 */
public class DenseLayer extends WeightLayer {
   public DenseLayer(int inputSize, int outputSize, Activation activation, WeightInitializer weightInitializer, double l1, double l2) {
      super(inputSize, outputSize, activation, weightInitializer, l1, l2);
   }

   public DenseLayer(WeightLayer layer) {
      super(layer);
   }

   public static Builder relu() {
      return new Builder().activation(Activation.RELU);
   }

   public static Builder sigmoid() {
      return new Builder().activation(Activation.SIGMOID);
   }

   @Override
   public Layer copy() {
      return new DenseLayer(this);
   }

   public static class Builder extends WeightLayerBuilder<DenseLayer.Builder> {

      @Override
      public Layer build() {
         return new DenseLayer(getInputSize(), getOutputSize(), getActivation(), getWeightInitializer(), getL1(),
                               getL2());
      }
   }

}// END OF DenseLayer
