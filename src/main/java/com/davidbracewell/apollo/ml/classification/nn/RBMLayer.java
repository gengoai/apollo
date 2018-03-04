package com.davidbracewell.apollo.ml.classification.nn;

import com.davidbracewell.apollo.linear.NDArray;
import com.davidbracewell.apollo.linear.NDArrayInitializer;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.data.Dataset;
import com.davidbracewell.apollo.ml.optimization.activation.Activation;

/**
 * @author David B. Bracewell
 */
public class RBMLayer extends WeightLayer {

   public RBMLayer(int inputSize, int outputSize) {
      super(inputSize, outputSize, Activation.SIGMOID,
            NDArrayInitializer.glorotAndBengioSigmoid(), 0, 0);
   }

   @Override
   public Layer copy() {
      return new RBMLayer(getInputSize(), getOutputSize());
   }

   @Override
   public void preTrain(Dataset<Instance> dataset) {
      NDArray weights = getWeights();

      // Do something to the weights
   }

}//END OF RBMLayer
