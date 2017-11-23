package com.davidbracewell.apollo.ml.classification.nn;


import com.davidbracewell.apollo.linear.NDArrayInitializer;
import com.davidbracewell.apollo.ml.optimization.activation.Activation;

/**
 * @author David B. Bracewell
 */
public class DenseLayer extends WeightLayer {
   public DenseLayer(int inputSize, int outputSize, Activation activation, NDArrayInitializer NDArrayInitializer, double l1, double l2) {
      super(inputSize, outputSize, activation, NDArrayInitializer, l1, l2);
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

   public static Builder linear() {
      return new Builder().activation(Activation.LINEAR);
   }


   @Override
   public Layer copy() {
      return new DenseLayer(this);
   }

   public static class Builder extends WeightLayerBuilder<DenseLayer.Builder> {

      @Override
      public Layer build() {
         return new DenseLayer(getInputSize(), getOutputSize(), getActivation(), this.getInitializer(), getL1(),
                               getL2());
      }
   }

}// END OF DenseLayer
