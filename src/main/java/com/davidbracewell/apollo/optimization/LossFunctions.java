package com.davidbracewell.apollo.optimization;

import com.davidbracewell.Math2;
import com.davidbracewell.apollo.linalg.Vector;

/**
 * @author David B. Bracewell
 */
public enum LossFunctions implements DifferentiableLossFunction {
   HINGE {
      @Override
      public double calculate(double p, double y) {
         y = y == 0 ? -1 : 1;
         p = p == 0 ? -1 : 1;
         return Math.max(0, 1 - y * p);
      }

      @Override
      public Vector gradient(Vector features, double p, double y) {
         y = y == 0 ? -1 : 1;
         p = p == 0 ? -1 : 1;
         if (y * p > 1) {
            return features.copy().zero();
         }
         return features.mapMultiply(y);
      }
   },
   LOGISTIC {
      @Override
      public double calculate(double p, double y) {
         p = Math2.clip(p, 1e-15, 1 - 1e-15);
         if (y == 1) {
            return -Math.log(p);
         }
         return -Math.log(1 - p);
      }

      @Override
      public Vector gradient(Vector features, double p, double y) {
         return features.mapMultiply(y - p);
      }
   },
   SQUARED {
      @Override
      public double calculate(double p, double y) {
         return 0.5 * Math.pow(p - y, 2);
      }

      @Override
      public Vector gradient(Vector features, double p, double y) {
         return features.mapMultiply(p - y);
      }
   }


}//END OF LossFunctions
