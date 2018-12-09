package com.gengoai.apollo.optimization;

import com.gengoai.apollo.linear.Axis;
import com.gengoai.apollo.linear.NDArray;
import com.gengoai.tuple.Tuple2;
import org.apache.commons.math3.util.FastMath;

import java.io.Serializable;

import static com.gengoai.tuple.Tuples.$;

/**
 * @author David B. Bracewell
 */
public class AdamUpdater implements WeightUpdate, Serializable {
   private static final long serialVersionUID = 1L;
   private double learningRate = 0.001;
   private double beta1 = 0.9;
   private double beta2 = 0.999;
   private double eps = 1e-08;
   private double decay = 0;

   private transient NDArray m = null;
   private transient NDArray v = null;

   @Override
   public WeightUpdate copy() {
      return null;
   }

   @Override
   public void reset() {
      m = null;
      v = null;
   }

   @Override
   public double update(LinearModelParameters weights,
                        GradientParameter gradient,
                        int iteration
                       ) {
      if (m == null) {
         m = weights.getWeights().getFactory().zeros(gradient.getWeightGradient().numRows(),
                                                     gradient.getWeightGradient().numCols());
      }
      if (v == null) {
         v = weights.getWeights().getFactory().zeros(gradient.getWeightGradient().numRows(),
                                                     gradient.getWeightGradient().numCols());
      }
      double addedCost = 0d;

      double lr = learningRate / (1.0 + decay * iteration);

      m = m.mul(beta1).add(gradient.getWeightGradient().mul(1d - beta1));
      v = v.mul(beta2).add(gradient.getWeightGradient().map(x -> (x * x) * (1 - beta2)));
      double lr_t = lr *
                       (
                          Math.sqrt(1.0 - FastMath.pow(beta2, iteration)) /
                             (1 - FastMath.pow(beta1, iteration))
                       );
      if (!Double.isFinite(lr_t) || lr_t == 0) {
         lr_t = eps;
      }

      weights.getWeights().subi(m.mul(lr_t).div(v.map(x -> Math.sqrt(x) + eps)));
      weights.getBias().subi(gradient.getBiasGradient().sum(Axis.ROW).muli(lr_t));
      return addedCost;
   }

   @Override
   public Tuple2<NDArray, Double> update(LinearModelParameters weights,
                                         NDArray input,
                                         NDArray output,
                                         NDArray delta,
                                         int iteration,
                                         boolean calculateOutDelta
                                        ) {
      if (m == null) {
         m = weights.getWeights().getFactory().zeros(output.numRows(), input.numRows());
      }
      if (v == null) {
         v = weights.getWeights().getFactory().zeros(output.numRows(), input.numRows());
      }
      double addedCost = 0d;


      double lr = learningRate / (1.0 + decay * iteration);

      NDArray dzOut = calculateOutDelta
                      ? weights.getWeights().T().mmul(delta)
                      : null;

      NDArray dw = delta.mmul(input.T())
                        .divi(input.numCols());

      m = m.mul(beta1).add(dw.mul(1d - beta1));
      v = v.mul(beta2).add(dw.map(x -> (x * x) * (1 - beta2)));
      double lr_t = lr *
                       (
                          Math.sqrt(1.0 - FastMath.pow(beta2, iteration)) /
                             (1 - FastMath.pow(beta1, iteration))
                       );
      if (!Double.isFinite(lr_t) || lr_t == 0) {
         lr_t = eps;
      }

      weights.getWeights().subi(m.mul(lr_t).div(v.map(x -> Math.sqrt(x) + eps)));
      NDArray db = delta.sum(Axis.ROW).divi(input.numCols());
      weights.getBias().subi(db.muli(lr_t));
      return $(dzOut, addedCost);
   }
}// END OF AdamUpdater
