package com.gengoai.apollo.ml.neural;

import com.gengoai.apollo.linear.NDArray;

/**
 * @author David B. Bracewell
 */
public class BackpropResult {
   public final NDArray delta;
   public final NDArray weightGradient;
   public final NDArray biasGradient;


   public BackpropResult(NDArray delta, NDArray weightGradient, NDArray biasGradient) {
      this.delta = delta;
      this.weightGradient = weightGradient;
      this.biasGradient = biasGradient;
   }


   public static BackpropResult from(NDArray delta, NDArray weightGradient, NDArray biasGradient) {
      return new BackpropResult(delta, weightGradient, biasGradient);
   }

}// END OF BackpropResult
