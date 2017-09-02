package com.davidbracewell.apollo.ml.classification.nn;

import com.davidbracewell.apollo.linalg.Matrix;
import com.davidbracewell.tuple.Tuple2;
import lombok.Builder;
import lombok.val;
import org.apache.commons.math3.util.FastMath;

import java.io.Serializable;

import static com.davidbracewell.tuple.Tuples.$;

/**
 * @author David B. Bracewell
 */
@Builder(toBuilder = true)
public class Adam implements WeightUpdate, Serializable {
   @Builder.Default
   private double learningRate = 0.001;
   @Builder.Default
   private double beta1 = 0.9;
   @Builder.Default
   private double beta2 = 0.999;
   @Builder.Default
   private double eps = 1e-08;
   @Builder.Default
   private double decay = 0;

   private transient Matrix m = null;
   private transient Matrix v = null;

   @Override
   public WeightUpdate copy() {
      return toBuilder().build();
   }

   @Override
   public double update(Matrix weights, Matrix bias, Matrix wGrad, Matrix bGrad, int iteration) {
      if (m == null) {
         m = weights.getFactory().zeros(wGrad.numRows(), wGrad.numCols());
      }
      if (v == null) {
         v = weights.getFactory().zeros(wGrad.numRows(), wGrad.numCols());
      }
      double addedCost = 0d;

      if (decay > 0) {
         learningRate *= 1.0 / (1.0 + decay * iteration);
      }

      m = m.mul(beta1).add(wGrad.mul(1d - beta1));
      v = v.mul(beta2).add(wGrad.map(x -> (x * x) * (1 - beta2)));
      double lr_t = learningRate *
                       (
                          Math.sqrt(1.0 - FastMath.pow(beta2, iteration)) /
                             (1 - FastMath.pow(beta1, iteration))
                       );
      if (!Double.isFinite(lr_t) || lr_t == 0) {
         lr_t = eps;
      }

      weights.subi(m.mul(lr_t).div(v.map(x -> Math.sqrt(x) + eps)));
      bias.subi(bGrad.muli(lr_t));
      return addedCost;
   }

   @Override
   public Tuple2<Matrix, Double> update(Matrix weights, Matrix bias, Matrix input, Matrix output, Matrix delta, int iteration, boolean calculateOutDelta) {
      if (m == null) {
         m = weights.getFactory().zeros(output.numRows(), input.numRows());
      }
      if (v == null) {
         v = weights.getFactory().zeros(output.numRows(), input.numRows());
      }
      double addedCost = 0d;

      if (decay > 0) {
         learningRate *= 1.0 / (1.0 + decay * iteration);
      }
      Matrix dzOut = calculateOutDelta
                     ? weights.transpose().mmul(delta)
                     : null;

      Matrix dw = delta.mmul(input.transpose())
                       .divi(input.numCols());


      m = m.mul(beta1).add(dw.mul(1d - beta1));
      v = v.mul(beta2).add(dw.map(x -> (x * x) * (1 - beta2)));

      double lr_t = learningRate *
                       (
                          Math.sqrt(1.0 - FastMath.pow(beta2, iteration)) /
                             (1 - FastMath.pow(beta1, iteration))
                       );
      if (!Double.isFinite(lr_t) || lr_t == 0) {
         lr_t = eps;
      }

      weights.subi(m.mul(lr_t).div(v.map(x -> Math.sqrt(x) + eps)));

      val db = delta.rowSums().divi(input.numCols());
      bias.subi(db.muli(lr_t));

      return $(dzOut, addedCost);
   }
}// END OF Adam
