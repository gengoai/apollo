package com.gengoai.apollo.optimization;


import com.gengoai.apollo.linear.p2.NDArray;
import com.gengoai.apollo.optimization.loss.LossFunction;

import java.util.Objects;

/**
 * @author David B. Bracewell
 */
public class GradientDescentCostFunction implements CostFunction<LinearModelParameters> {
   LossFunction lossFunction;
   int trueLabel;

   public GradientDescentCostFunction(LossFunction lossFunction) {
      this(lossFunction, -1);
   }

   public GradientDescentCostFunction(LossFunction lossFunction, int trueLabel) {
      this.lossFunction = lossFunction;
      this.trueLabel = trueLabel;
   }

   @Override
   public int hashCode() {
      return Objects.hash(lossFunction, trueLabel);
   }

   @Override
   public boolean equals(Object obj) {
      if (this == obj) {return true;}
      if (obj == null || getClass() != obj.getClass()) {return false;}
      final GradientDescentCostFunction other = (GradientDescentCostFunction) obj;
      return Objects.equals(this.lossFunction, other.lossFunction)
                && Objects.equals(this.trueLabel, other.trueLabel);
   }

   @Override
   public CostGradientTuple evaluate(NDArray vector, LinearModelParameters theta) {
      NDArray predicted = theta.activate(vector);
      NDArray y = vector.getLabelAsNDArray(theta.getNumberOfWeightVectors());
      if (theta.isBinary() && y.rows() > 1) {
         y = y.getRow(trueLabel);
      }
      NDArray derivative = lossFunction.derivative(predicted, y);
      return CostGradientTuple.of(lossFunction.loss(predicted, y),
                                  GradientParameter.calculate(vector, derivative),
                                  new NDArray[]{predicted});
   }

   public LossFunction getLossFunction() {
      return this.lossFunction;
   }

   public int getTrueLabel() {
      return this.trueLabel;
   }

   public String toString() {
      return "GradientDescentCostFunction(lossFunction=" + this.getLossFunction() + ", trueLabel=" + this.getTrueLabel() + ")";
   }
}//END OF GradientDescentCostFunction
