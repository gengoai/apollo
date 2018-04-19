package com.gengoai.apollo.ml.optimization.activation;

import com.gengoai.apollo.linear.Axis;
import com.gengoai.apollo.linear.NDArray;
import lombok.val;
import org.apache.commons.math3.util.FastMath;

/**
 * @author David B. Bracewell
 */
public class SoftmaxActivation implements Activation {
   private static final long serialVersionUID = 1L;

   @Override
   public double apply(double x) {
      return SIGMOID.apply(x);
   }

   @Override
   public NDArray apply(NDArray x) {
      if (x.isVector()) {
         val max = x.max();
         x.mapi(d -> FastMath.exp(d - max));
         val sum = x.sum();
         return x.divi(sum);
      }
      val max = x.max(Axis.COlUMN);
      x.mapi(max, Axis.ROW, (d1, m) -> FastMath.exp(d1 - m));
      val sum = x.sum(Axis.COlUMN);
      return x.divi(sum, Axis.ROW);
   }

   @Override
   public boolean isProbabilistic() {
      return true;
   }

   @Override
   public double valueGradient(double activated) {
      return activated * (1d - activated);
   }

}// END OF SoftmaxActivation
