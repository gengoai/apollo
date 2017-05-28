package com.davidbracewell.apollo.optimization;

import java.io.Serializable;

/**
 * @author David B. Bracewell
 */
public class Softmax implements Activation, Serializable {
   @Override
   public double apply(double x) {

      return 0;
   }

   @Override
   public double gradient(double x) {
      return 0;
   }

   @Override
   public double valueGradient(double x) {
      return 0;
   }
}// END OF Softmax
