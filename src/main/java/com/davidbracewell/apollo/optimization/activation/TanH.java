package com.davidbracewell.apollo.optimization.activation;

import com.davidbracewell.apollo.linalg.Matrix;
import lombok.val;
import org.apache.commons.math3.util.FastMath;

/**
 * @author David B. Bracewell
 */
public class TanH implements Activation {

   @Override
   public double apply(double x) {
      double ez = FastMath.exp(x);
      double enz = FastMath.exp(-x);
      return (ez - enz) / (ez + enz);
   }


   @Override
   public Matrix apply(Matrix m) {
      val ez = m.exp();
      val ezn = m.neg().exp();
      return (ez.sub(ezn)).div(ez.add(ezn));
   }

   @Override
   public Matrix valueGradient(Matrix m) {
      return m.map(this::valueGradient);
   }

   @Override
   public double valueGradient(double activated) {
      return 1.0 - (activated * activated);
   }

}// END OF TanH
