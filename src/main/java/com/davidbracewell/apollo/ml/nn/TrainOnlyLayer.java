package com.davidbracewell.apollo.ml.nn;

/**
 * @author David B. Bracewell
 */
public abstract class TrainOnlyLayer extends Layer {
   private static final long serialVersionUID = 1L;

   public TrainOnlyLayer(int inputSize, int outputSize) {
      super(inputSize, outputSize);
   }
}// END OF TrainOnlyLayer
