package com.gengoai.apollo.ml.optimization;

/**
 * @author David B. Bracewell
 */

import com.gengoai.apollo.linear.Axis;
import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.ml.optimization.loss.LossFunction;
import lombok.Value;

/**
 * @author David B. Bracewell
 */
@Value
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
   public CostGradientTuple evaluate(NDArray vector, LinearModelParameters theta) {
      NDArray predicted = theta.activate(vector);
      NDArray y = vector.getLabelAsNDArray(theta.getNumberOfWeightVectors());
      if (theta.isBinary() && y.numRows() > 1) {
         y = y.getVector(trueLabel, Axis.ROW);
      }
      NDArray derivative = lossFunction.derivative(predicted, y);
      return CostGradientTuple.of(lossFunction.loss(predicted, y),
                                  GradientParameter.calculate(vector, derivative),
                                  new NDArray[]{predicted});
   }
}//END OF GradientDescentCostFunction
