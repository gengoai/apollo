package com.davidbracewell.apollo.optimization;

import lombok.Value;

/**
 * @author David B. Bracewell
 */
@Value(staticConstructor = "of")
public class CostGradientTuple {
   double loss;
   Gradient[] gradients;

   public Gradient getGradient() {
      return gradients[0];
   }

   public Gradient getGradient(int index) {
      return gradients[index];
   }

   public static CostGradientTuple of(double loss, Gradient gradient) {
      return of(loss, new Gradient[]{gradient});
   }

}// END OF CostGradientTuple
