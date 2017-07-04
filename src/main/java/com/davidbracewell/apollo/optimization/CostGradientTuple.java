package com.davidbracewell.apollo.optimization;

import lombok.Value;

/**
 * @author David B. Bracewell
 */
@Value(staticConstructor = "of")
public class CostGradientTuple {
   double loss;
   Gradient[] gradients;

   public static CostGradientTuple of(double loss, Gradient gradient) {
      return of(loss, new Gradient[]{gradient});
   }

   public Gradient getGradient() {
      return gradients[0];
   }

   public Gradient getGradient(int index) {
      if (index > gradients.length && gradients.length == 1) {
         return gradients[0];
      }
      return gradients[index];
   }

}// END OF CostGradientTuple
