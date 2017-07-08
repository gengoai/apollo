package com.davidbracewell.apollo.optimization.alt;

import com.davidbracewell.apollo.linalg.Vector;
import lombok.Getter;

import java.io.Serializable;

/**
 * @author David B. Bracewell
 */

public class WeightVector implements Serializable {
   @Getter
   private Vector weights;
   @Getter
   private double bias = 0;


   public WeightVector(int dimension) {
      this.weights = Vector.sZeros(dimension);
   }

   public double dot(Vector v) {
      return weights.dot(v) + bias;
   }

   public void update(Gradient gradient) {
      weights.subtractSelf(gradient.getWeightGradient());
      bias -= gradient.getBiasGradient();
   }


}//END OF WeightVector
