package com.davidbracewell.apollo.optimization.alt;

import com.davidbracewell.apollo.linalg.Vector;

import java.io.Serializable;

/**
 * @author David B. Bracewell
 */
 public class GradientMatrix implements Serializable {
   Gradient[] gradients;

   public GradientMatrix(Vector input, Vector delta) {
      gradients = new Gradient[delta.dimension()];
      for (int i = 0; i < delta.dimension(); i++) {
         gradients[i] = Gradient.of(Vector.sZeros(input.dimension()), delta.get(i));
      }
      for (int r = 0; r < input.dimension(); r++) {
         for (int c = 0; c < delta.dimension(); c++) {
            gradients[c].getWeightGradient()
                        .increment(r, input.get(r) * delta.get(c));
         }
      }
   }

   public GradientMatrix(Gradient gradient) {
      this.gradients = new Gradient[]{gradient};
   }

}// END OF GradientMatrix
