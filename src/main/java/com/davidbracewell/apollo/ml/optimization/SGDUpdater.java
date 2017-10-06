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

   protected double l1Update(NDArray weights, double learningRate, int iteration) {
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

   protected double l2Update(NDArray gradient) {
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
   public Tuple2<Double, NDArray> update(LinearModelParameters weights, NDArray input, NDArray output, NDArray delta, int iteration, boolean calculateOutDelta) {
      return null;
   }

   @Override
   public double update(LinearModelParameters weights, GradientParameter gradient, int iteration) {
      if (momentum > 0 && v == null) {
         v = weights.getWeights().getFactory().zeros(weights.getWeights().shape());
      }
      double lr = learningRate / (1.0 + decayRate * iteration);
      double addedCost = 0;
      addedCost += l2Update(gradient.getWeightGradient());

      if (momentum > 0) {
         v = v.muli(momentum).subi(gradient.getWeightGradient().muli(lr));
         weights.getWeights().addi(v);
      } else {
         weights.getWeights().subi(gradient.getWeightGradient().muli(lr));
      }

      weights.getBias().subi(gradient.getBiasGradient().sum(Axis.ROW).muli(lr));

      addedCost += l1Update(weights.getWeights(), lr, iteration);
      return addedCost;
   }
}// END OF SGDUpdater
