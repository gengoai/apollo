package com.davidbracewell.apollo.ml.nn;

import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.optimization.activation.DifferentiableActivation;
import lombok.NonNull;

/**
 * @author David B. Bracewell
 */
public class ActivationLayer implements Layer {
   private static final long serialVersionUID = 1L;
   protected final DifferentiableActivation activation;
   protected int outputSize;
   private int inputSize;

   public ActivationLayer(@NonNull DifferentiableActivation activation) {
      this.activation = activation;
   }

   @Override
   public Vector backward(Vector output, Vector delta) {
      return delta.multiply(activation.valueGradient(output));
   }

   public static LayerBuilder builder(@NonNull DifferentiableActivation activation) {
      return new ActivationBuilder(activation);
   }

   @Override
   public void connect(Layer previousLayer) {
      this.inputSize = previousLayer.getOutputSize();
      this.outputSize = previousLayer.getOutputSize();
   }

   @Override
   public Vector forward(Vector input) {
      return activation.apply(input);
   }

   @Override
   public int getInputSize() {
      return inputSize;
   }

   @Override
   public int getOutputSize() {
      return outputSize;
   }

   @Override
   public boolean hasWeights() {
      return false;
   }

   private static class ActivationBuilder extends LayerBuilder {
      private final DifferentiableActivation activation;

      private ActivationBuilder(DifferentiableActivation activation) {
         this.activation = activation;
      }

      @Override
      public Layer build() {
         return new ActivationLayer(activation);
      }
   }

}// END OF ActivationLayer
