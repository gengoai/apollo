package com.davidbracewell.apollo.ml.nn.n2;

import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.optimization.update.WeightUpdate;

/**
 * @author David B. Bracewell
 */
public class InputLayer implements Layer {
   private final int inputSize;

   public InputLayer(int inputSize) {
      this.inputSize = inputSize;
   }

   @Override
   public Vector backward(Vector input, Vector output, Vector delta, WeightUpdate weightUpdate, double lr) {
      return delta;
   }

   @Override
   public void connect(Layer previousLayer) {

   }

   @Override
   public Vector forward(Vector input) {
      return input;
   }

   @Override
   public int getInputSize() {
      return inputSize;
   }

   @Override
   public int getOutputSize() {
      return inputSize;
   }

   @Override
   public boolean isOptimizable() {
      return false;
   }

}// END OF InputLayer
