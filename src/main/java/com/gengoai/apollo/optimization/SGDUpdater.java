package com.gengoai.apollo.optimization;

import com.gengoai.apollo.linear.Axis;
import com.gengoai.apollo.linear.NDArray;
import com.gengoai.concurrent.AtomicDouble;
import com.gengoai.tuple.Tuple2;
import org.apache.commons.math3.util.FastMath;

import java.io.Serializable;

import static com.gengoai.tuple.Tuples.$;

/**
 * @author David B. Bracewell
 */
public class SGDUpdater implements WeightUpdate, Serializable {
   private static final long serialVersionUID = 1L;
   private double learningRate = 0.01;
   private double decayRate = 0.01;
   private double momentum = 0.90;
   private double l1 = 0.0;
   private double l2 = 0.0;
   private transient NDArray v;

   @java.beans.ConstructorProperties({"learningRate", "decayRate", "momentum", "l1", "l2", "v"})
   public SGDUpdater(double learningRate, double decayRate, double momentum, double l1, double l2, NDArray v) {
      this.learningRate = learningRate;
      this.decayRate = decayRate;
      this.momentum = momentum;
      this.l1 = l1;
      this.l2 = l2;
      this.v = v;
   }

   public static SGDUpdaterBuilder builder() {
      return new SGDUpdaterBuilder();
   }

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
            double xp = FastMath.signum(x) * FastMath.max(0, FastMath.abs(x) - shrinkage);
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

   public SGDUpdaterBuilder toBuilder() {
      return new SGDUpdaterBuilder().learningRate(this.learningRate).decayRate(this.decayRate).momentum(
         this.momentum).l1(this.l1).l2(this.l2).v(this.v);
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

      NDArray dw = delta.mmul(input.T())
                    .divi(input.numCols());
      NDArray db = delta.sum(Axis.ROW)
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

      private double learningRate = 0.01;
      private double decayRate = 0.01;
      private double momentum = 0.90;
      private double l1 = 0.0;
      private double l2 = 0.0;
      private NDArray v;

      SGDUpdaterBuilder() {
      }

      public SGDUpdater build() {
         return new SGDUpdater(learningRate, decayRate, momentum, l1, l2, v);
      }

      public SGDUpdaterBuilder decayRate(double decayRate) {
         this.decayRate = decayRate;
         return this;
      }

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

      public SGDUpdaterBuilder l1(double l1) {
         this.l1 = l1;
         return this;
      }

      public SGDUpdaterBuilder l2(double l2) {
         this.l2 = l2;
         return this;
      }

      public SGDUpdaterBuilder learningRate(double learningRate) {
         this.learningRate = learningRate;
         return this;
      }

      public SGDUpdaterBuilder momentum(double momentum) {
         this.momentum = momentum;
         return this;
      }

      public String toString() {
         return "SGDUpdater.SGDUpdaterBuilder(decayRate=" + this.getDecayRate() + ", l1=" + this.getL1() + ", l2=" + this
                                                                                                                        .getL2() + ", learningRate=" + this
                                                                                                                                                          .getLearningRate() + ", momentum=" + this
                                                                                                                                                                                                  .getMomentum() + ", v=" + this.v + ")";
      }

      public SGDUpdaterBuilder v(NDArray v) {
         this.v = v;
         return this;
      }
   }

}// END OF SGDUpdater
