package com.davidbracewell.apollo.ml.optimization.activation;

import com.davidbracewell.apollo.linear.NDArray;

/**
 * The type Linear function.
 *
 * @author David B. Bracewell
 */
public class LinearActivation implements Activation {
   private static final long serialVersionUID = 1L;

   @Override
   public double apply(double x) {
      return x;
   }

   @Override
   public NDArray apply(NDArray x) {
      return x;
   }

   @Override
   public NDArray gradient(NDArray in) {
      return in.getFactory().ones(in.numRows(), in.numCols());
   }

   @Override
   public double valueGradient(double activated) {
      return 1;
   }

   @Override
   public NDArray valueGradient(NDArray activated) {
      return activated.getFactory().ones(activated.numRows(), activated.numCols());
   }
}// END OF LinearActivation
