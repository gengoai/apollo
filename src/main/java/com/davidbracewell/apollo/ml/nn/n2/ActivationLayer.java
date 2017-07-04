package com.davidbracewell.apollo.ml.nn.n2;

import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.optimization.activation.DifferentiableActivation;
import com.davidbracewell.apollo.optimization.update.WeightUpdate;
import lombok.NonNull;

/**
 * @author David B. Bracewell
 */
public class ActivationLayer implements Layer {
   private final DifferentiableActivation activation;
   private int inputSize;
   private int outputSize;

   public ActivationLayer(@NonNull DifferentiableActivation activation) {
      this.activation = activation;
   }

   @Override
   public Vector backward(Vector input, Vector output, Vector delta, WeightUpdate weightUpdate, double lr) {
      return delta
                .multiply(activation.valueGradient(output));
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
   public void connect(Layer previousLayer) {
      this.inputSize = previousLayer.getInputSize();
      this.outputSize = previousLayer.getOutputSize();
   }

   @Override
   public boolean isOptimizable() {
      return false;
   }

}// END OF ActivationLayer
