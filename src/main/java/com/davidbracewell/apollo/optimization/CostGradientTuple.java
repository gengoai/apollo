package com.davidbracewell.apollo.optimization;

import com.davidbracewell.apollo.linalg.Vector;
import lombok.Value;

/**
 * @author David B. Bracewell
 */
@Value(staticConstructor = "of")
public class CostGradientTuple {
   double loss;
   Vector[] gradients;

   public Vector getGradient() {
      return gradients[0];
   }

   public Vector getGradient(int index) {
      return gradients[index];
   }

   public static CostGradientTuple of(double loss, Vector gradient) {
      return of(loss, new Vector[]{gradient});
   }

}// END OF CostGradientTuple
