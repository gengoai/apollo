package com.davidbracewell.apollo.ml.nn;

import com.davidbracewell.apollo.linalg.Vector;

/**
 * @author David B. Bracewell
 */
public class InputLayer implements Layer {
   private static final long serialVersionUID = 1L;
   private final int inputSize;

   public InputLayer(int inputSize) {
      this.inputSize = inputSize;
   }


   @Override
   public Vector backward(Vector output, Vector delta) {
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
   public boolean hasWeights() {
      return false;
   }

}// END OF InputLayer
