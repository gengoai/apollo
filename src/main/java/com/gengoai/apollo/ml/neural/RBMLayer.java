package com.gengoai.apollo.ml.neural;

import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.linear.NDArrayInitializer;
import com.gengoai.apollo.ml.data.Dataset;
import com.gengoai.apollo.optimization.activation.Activation;

/**
 * @author David B. Bracewell
 */
public class RBMLayer extends WeightLayer {

   public RBMLayer(int inputSize, int outputSize) {
      super(inputSize, outputSize, Activation.SIGMOID,
            NDArrayInitializer.glorotAndBengioSigmoid, 0, 0);
   }

   @Override
   public Layer copy() {
      return new RBMLayer(getInputSize(), getOutputSize());
   }

   @Override
   public void preTrain(Dataset dataset) {
      NDArray weights = getWeights();

      // Do something to the weights
   }

}//END OF RBMLayer
