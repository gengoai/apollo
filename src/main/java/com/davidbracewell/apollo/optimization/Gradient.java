package com.davidbracewell.apollo.optimization;

import com.davidbracewell.apollo.linalg.Vector;
import lombok.Data;

/**
 * @author David B. Bracewell
 */
@Data
public class Gradient {
   private Vector weightGradient;
   private double biasGradient;

   public static Gradient of(Vector weightGradient, double biasGradient) {
      Gradient g = new Gradient();
      g.weightGradient = weightGradient;
      g.biasGradient = biasGradient;
      return g;
   }

   public void scale(double scale) {
      weightGradient.mapMultiplySelf(scale);
      biasGradient *= scale;
   }

}//END OF Gradient
