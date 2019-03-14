package com.gengoai.apollo.optimization.activation;

import com.gengoai.apollo.linear.NDArray;

import static com.gengoai.apollo.linear.NDArrayFactory.ND;

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
      return ND.ones(in.rows(), in.columns());
   }

   @Override
   public double valueGradient(double activated) {
      return 1;
   }

   @Override
   public NDArray valueGradient(NDArray activated) {
      return ND.ones(activated.rows(), activated.columns());
   }
}// END OF LinearActivation
