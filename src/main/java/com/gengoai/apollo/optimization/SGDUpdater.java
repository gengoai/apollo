package com.gengoai.apollo.optimization;

import com.gengoai.apollo.linear.NDArray;
import com.gengoai.concurrent.AtomicDouble;
import com.gengoai.tuple.Tuple2;
import org.apache.commons.math3.util.FastMath;

import java.io.Serializable;

import static com.gengoai.apollo.linear.NDArrayFactory.ND;
import static com.gengoai.tuple.Tuples.$;

/**
 * The type Sgd updater.
 *
 * @author David B. Bracewell
 */
public class SGDUpdater implements WeightUpdate, Serializable {
   private static final long serialVersionUID = 1L;
   private double learningRate;
   private double decayRate;
   private double momentum;
   private double l1;
   private double l2;
   private transient NDArray v;

   /**
    * Instantiates a new Sgd updater.
    *
    * @param learningRate the learning rate
    * @param decayRate    the decay rate
    * @param momentum     the momentum
    * @param l1           the l 1
    * @param l2           the l 2
    * @param v            the v
    */
   public SGDUpdater(double learningRate, double decayRate, double momentum, double l1, double l2, NDArray v) {
      this.learningRate = learningRate;
      this.decayRate = decayRate;
      this.momentum = momentum;
      this.l1 = l1;
      this.l2 = l2;
      this.v = v;
   }

   /**
    * Builder sgd updater builder.
    *
    * @return the sgd updater builder
    */
   public static SGDUpdaterBuilder builder() {
      return new SGDUpdaterBuilder();
   }

   @Override
   public WeightUpdate copy() {
      return toBuilder().build();
   }

   /**
    * L 1 update double.
    *
    * @param weights      the weights
    * @param learningRate the learning rate
    * @param l1           the l 1
    * @param iteration    the iteration
    * @return the double
    */
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

   /**
    * L 2 update double.
    *
    * @param gradient the gradient
    * @param l2       the l 2
    * @return the double
    */
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

   /**
    * To builder sgd updater builder.
    *
    * @return the sgd updater builder
    */
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
         v = ND.array(output.rows(), input.rows());
      }
      double lr = learningRate / (1.0 + decayRate * iteration);

      double addedCost = 0;
      NDArray dzOut = calculateOutDelta
                      ? weights.getWeights().T().mmul(delta)
                      : null;

      NDArray dw = delta.mmul(input.T())
                        .divi(input.columns());
      NDArray db = delta.rowSums()
                        .divi(input.columns());

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
         v = ND.array(weights.getWeights().rows(), weights.getWeights().columns());
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


      weights.getBias().subi(gradient.getBiasGradient().rowSums().muli(lr));
      addedCost += l1Update(weights.getWeights(), lr, l1, iteration);
      return addedCost;
   }

   /**
    * The type Sgd updater builder.
    */
   public static class SGDUpdaterBuilder {

      private double learningRate = 0.01;
      private double decayRate = 0.01;
      private double momentum = 0.90;
      private double l1 = 0.0;
      private double l2 = 0.0;
      private NDArray v;

      /**
       * Instantiates a new Sgd updater builder.
       */
      SGDUpdaterBuilder() {
      }

      /**
       * Build sgd updater.
       *
       * @return the sgd updater
       */
      public SGDUpdater build() {
         return new SGDUpdater(learningRate, decayRate, momentum, l1, l2, v);
      }

      /**
       * Decay rate sgd updater builder.
       *
       * @param decayRate the decay rate
       * @return the sgd updater builder
       */
      public SGDUpdaterBuilder decayRate(double decayRate) {
         this.decayRate = decayRate;
         return this;
      }

      /**
       * Gets learning rate.
       *
       * @return the learning rate
       */
      public double getLearningRate() {
         return learningRate;
      }

      /**
       * Gets decay rate.
       *
       * @return the decay rate
       */
      public double getDecayRate() {
         return decayRate;
      }

      /**
       * Gets momentum.
       *
       * @return the momentum
       */
      public double getMomentum() {
         return momentum;
      }

      /**
       * Gets l 1.
       *
       * @return the l 1
       */
      public double getL1() {
         return l1;
      }

      /**
       * Gets l 2.
       *
       * @return the l 2
       */
      public double getL2() {
         return l2;
      }

      /**
       * L 1 sgd updater builder.
       *
       * @param l1 the l 1
       * @return the sgd updater builder
       */
      public SGDUpdaterBuilder l1(double l1) {
         this.l1 = l1;
         return this;
      }

      /**
       * L 2 sgd updater builder.
       *
       * @param l2 the l 2
       * @return the sgd updater builder
       */
      public SGDUpdaterBuilder l2(double l2) {
         this.l2 = l2;
         return this;
      }

      /**
       * Learning rate sgd updater builder.
       *
       * @param learningRate the learning rate
       * @return the sgd updater builder
       */
      public SGDUpdaterBuilder learningRate(double learningRate) {
         this.learningRate = learningRate;
         return this;
      }

      /**
       * Momentum sgd updater builder.
       *
       * @param momentum the momentum
       * @return the sgd updater builder
       */
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

      /**
       * V sgd updater builder.
       *
       * @param v the v
       * @return the sgd updater builder
       */
      public SGDUpdaterBuilder v(NDArray v) {
         this.v = v;
         return this;
      }
   }

}// END OF SGDUpdater
