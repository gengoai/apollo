package com.davidbracewell.apollo.ml.nn;

/**
 * @author David B. Bracewell
 */
public abstract class BaseLayer implements Layer {
   private static final long serialVersionUID = 1L;
   private final int inputSize;
   private final int outputSize;

   public BaseLayer(int inputSize, int outputSize) {
      this.inputSize = inputSize;
      this.outputSize = outputSize;
   }

   @Override
   public int getInputSize() {
      return inputSize;
   }

   @Override
   public int getOutputSize() {
      return outputSize;
   }

}// END OF BaseLayer
