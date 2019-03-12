package com.gengoai.apollo.optimization;

import com.gengoai.apollo.linear.p2.NDArray;

/**
 * @author David B. Bracewell
 */
public class CostGradientTuple {
   public double getCost() {
      return cost;
   }

   public GradientParameter getGradient() {
      return gradient;
   }

   public NDArray[] getActivations() {
      return activations;
   }

   public final double cost;
   public final GradientParameter gradient;
   public final NDArray[] activations;


   protected CostGradientTuple(double cost, GradientParameter gradient, NDArray[] activations) {
      this.cost = cost;
      this.gradient = gradient;
      this.activations = activations;
   }

   public static CostGradientTuple of(double cost, GradientParameter gradient, NDArray[] activations) {
      return new CostGradientTuple(cost, gradient, activations);
   }

}// END OF CostGradientTuple
