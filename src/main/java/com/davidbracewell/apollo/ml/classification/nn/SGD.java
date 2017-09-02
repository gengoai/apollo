package com.davidbracewell.apollo.ml.classification.nn;

import com.davidbracewell.apollo.linalg.Matrix;
import com.davidbracewell.guava.common.util.concurrent.AtomicDouble;
import com.davidbracewell.tuple.Tuple2;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.val;
import org.apache.commons.math3.util.FastMath;

import java.io.Serializable;

import static com.davidbracewell.tuple.Tuples.$;

/**
 * @author David B. Bracewell
 */
@Builder(toBuilder = true)
@AllArgsConstructor
public class SGD implements WeightUpdate, Serializable {
   private static final long serialVersionUID = 1L;
   @Builder.Default
   private double learningRate = 0.01;
   @Builder.Default
   private double decayRate = 0.01;
   @Builder.Default
   private double momentum = 0.90;
   @Builder.Default
   private double l1 = 0.0;
   @Builder.Default
   private double l2 = 0.0;
   private transient Matrix v;

   @Override
   public WeightUpdate copy() {
      return toBuilder().build();
   }

   protected double l1Update(Matrix weights, double learningRate, int iteration) {
      if (l1 > 0) {
         AtomicDouble cost = new AtomicDouble(0);
         double shrinkage = l1 * (learningRate / iteration);
         weights.mapi(x -> {
            cost.addAndGet(FastMath.abs(x));
            val xp = FastMath.signum(x) * FastMath.max(0, FastMath.abs(x) - shrinkage);
            if (FastMath.abs(xp) < 1e-9) {
               return 0d;
            }
            return xp;
         });
         return cost.get();
      }
      return 0;
   }

   protected double l2Update(Matrix gradient) {
      if (l2 > 0) {
         AtomicDouble addedCost = new AtomicDouble(0d);
         gradient.mapi(x -> {
            double square = x * x;
            addedCost.addAndGet(square);
            return x * l2;
         });
         return l2 * addedCost.get() / 2d;
      }
      return 0;
   }

   @Override
   public double update(Matrix weights, Matrix bias, Matrix wGrad, Matrix bGrad, int iteration) {
      if (momentum > 0 && v == null) {
         v = weights.getFactory().zeros(weights.numRows(), weights.numCols());
      }
      double lr = learningRate / (1.0 + decayRate * iteration);
      double addedCost = 0;
      addedCost += l2Update(wGrad);

      if (momentum > 0) {
         v = v.muli(momentum).subi(wGrad.muli(lr));
         weights.addi(v);
      } else {
         weights.subi(wGrad.muli(lr));
      }

      bias.subi(bGrad.muli(lr));

      addedCost += l1Update(weights, lr, iteration);
      return addedCost;
   }

   @Override
   public Tuple2<Matrix, Double> update(Matrix weights, Matrix bias, Matrix input, Matrix output, Matrix delta, int iteration, boolean calculateOutDelta) {
      if (momentum > 0 && v == null) {
         v = weights.getFactory().zeros(output.numRows(), input.numRows());
      }
      double lr = learningRate / (1.0 + decayRate * iteration);

      double addedCost = 0;
      Matrix dzOut = calculateOutDelta
                     ? weights.transpose().mmul(delta)
                     : null;

      val dw = delta.mmul(input.transpose())
                    .divi(input.numCols());
      val db = delta.rowSums()
                    .divi(input.numCols());

      addedCost += l2Update(dw);

      if (momentum > 0) {
         v = v.muli(momentum).subi(dw.muli(lr));
         weights.addi(v);
      } else {
         weights.subi(dw.muli(lr));
      }

      bias.subi(db.muli(lr));

      addedCost += l1Update(weights, lr, iteration);


      return $(dzOut, addedCost);
   }

}// END OF SGD
