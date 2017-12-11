package com.davidbracewell.apollo.ml.optimization;

import com.davidbracewell.apollo.linear.Axis;
import com.davidbracewell.apollo.linear.NDArray;
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
public class SGDUpdater implements WeightUpdate, Serializable {
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
   private transient NDArray v;

   @Override
   public WeightUpdate copy() {
      return toBuilder().build();
   }

   public static double l1Update(NDArray weights, double learningRate, double l1, int iteration) {
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

   public static double l2Update(NDArray gradient, double l2) {
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
   public void reset() {
      v = null;
   }

   @Override
   public Tuple2<NDArray, Double> update(LinearModelParameters weights,
                                         NDArray input,
                                         NDArray output,
                                         NDArray delta,
                                         int iteration,
                                         boolean calculateOutDelta
                                        ) {
      if (momentum > 0 && v == null) {
         v = weights.getWeights().getFactory().zeros(output.numRows(), input.numRows());
      }
      double lr = learningRate / (1.0 + decayRate * iteration);

      double addedCost = 0;
      NDArray dzOut = calculateOutDelta
                      ? weights.getWeights().T().mmul(delta)
                      : null;

      val dw = delta.mmul(input.T())
                    .divi(input.numCols());
      val db = delta.sum(Axis.ROW)
                    .divi(input.numCols());

      addedCost += l2Update(dw, l2);

      if (momentum > 0) {
         v = v.muli(momentum).subi(dw.muli(lr));
         weights.getWeights().addi(v);
      } else {
         weights.getWeights().subi(dw.muli(lr));
      }

      weights.getBias().subi(db.muli(lr));

      addedCost += l1Update(weights.getWeights(), lr, l1, iteration);

      return $(dzOut, addedCost);
   }

   @Override
   public double update(LinearModelParameters weights, GradientParameter gradient, int iteration) {
      if (momentum > 0 && v == null) {
         v = weights.getWeights().getFactory().zeros(weights.getWeights().numRows(),
                                                     weights.getWeights().numCols());
      }
      double lr = learningRate / (1.0 + decayRate * iteration);
      double addedCost = 0;
      addedCost += l2Update(gradient.getWeightGradient(), l2);

      if (momentum > 0) {
         v = v.muli(momentum).subi(gradient.getWeightGradient().muli(lr));
         weights.getWeights().addi(v);
      } else {
         weights.getWeights().subi(gradient.getWeightGradient().muli(lr));
      }

      weights.getBias().subi(gradient.getBiasGradient().sum(Axis.ROW).muli(lr));

      addedCost += l1Update(weights.getWeights(), lr, l1, iteration);
      return addedCost;
   }

   public static class SGDUpdaterBuilder {

      public double getLearningRate() {
         return learningRate;
      }

      public double getDecayRate() {
         return decayRate;
      }

      public double getMomentum() {
         return momentum;
      }

      public double getL1() {
         return l1;
      }

      public double getL2() {
         return l2;
      }
   }

}// END OF SGDUpdater
