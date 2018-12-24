package com.gengoai.apollo.ml.neural;

import com.gengoai.apollo.linear.Axis;
import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.optimization.CostFunction;
import com.gengoai.apollo.optimization.CostGradientTuple;
import com.gengoai.apollo.optimization.GradientParameter;
import com.gengoai.apollo.optimization.loss.LossFunction;

/**
 * @author David B. Bracewell
 */
public class FeedForwardCostFunction implements CostFunction<FeedForwardNetwork> {
   LossFunction lossFunction;

   public FeedForwardCostFunction(LossFunction lossFunction) {
      this.lossFunction = lossFunction;
   }

   @Override
   public CostGradientTuple evaluate(NDArray input, FeedForwardNetwork network) {
      NDArray[] ai = new NDArray[network.layers.size()];
      NDArray cai = input;
      NDArray Y = input.getLabelAsNDArray();
      for (int i = 0; i < network.layers.size(); i++) {
         cai = network.layers.get(i).forward(cai);
         ai[i] = cai;
      }
      if (cai.numRows() == 1) { //If Binary, only take the first row of the Y
         Y = Y.getVector(1, Axis.ROW);
      }
      double loss = lossFunction.loss(cai, Y) / input.numCols();
      NDArray dz = lossFunction.derivative(cai, Y);
      return CostGradientTuple.of(loss,
                                  GradientParameter.of(dz, dz),
                                  ai);
   }
}// END OF FeedForwardCostFunction
