package com.davidbracewell.apollo.optimization;

import com.davidbracewell.apollo.linalg.Matrix;
import com.davidbracewell.apollo.linalg.Vector;
import lombok.NonNull;
import lombok.Value;

/**
 * @author David B. Bracewell
 */
@Value(staticConstructor = "of")
public class Gradient {
   Matrix weightGradient;
   Vector biasGradient;

   public Gradient addSelf(@NonNull Gradient other) {
      this.weightGradient.addSelf(other.weightGradient);
      this.biasGradient.addSelf(other.biasGradient);
      return this;
   }

   public Gradient mapDivideSelf(@NonNull double number) {
      this.weightGradient.scaleSelf(1d / number);
      this.biasGradient.mapDivideSelf(number);
      return this;
   }

   public Gradient respectToInput(@NonNull Vector vector) {
      return Gradient.of(this.weightGradient.multiply(vector.toMatrix()), biasGradient);
   }

   public Gradient scaleSelf(double scale) {
      this.weightGradient.scaleSelf(scale);
      this.biasGradient.mapMultiplySelf(scale);
      return this;
   }

}//END OF Gradient
