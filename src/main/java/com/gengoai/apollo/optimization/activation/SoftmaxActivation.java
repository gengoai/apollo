package com.gengoai.apollo.optimization.activation;

import com.gengoai.apollo.linear.p2.NDArray;
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
      if (x.shape().isVector()) {
         double max = x.max();
         x.mapi(d -> FastMath.exp(d - max));
         return x.divi(x.sum());
      }
      NDArray max = x.columnMaxs();
      x.mapiRow(max, (d1, m) -> FastMath.exp(d1 - m));
      NDArray sum = x.columnSums();
      return x.diviRowVector(sum);
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
