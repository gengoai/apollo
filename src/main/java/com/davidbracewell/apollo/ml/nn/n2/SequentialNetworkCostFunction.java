package com.davidbracewell.apollo.ml.nn.n2;

import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.optimization.CostFunction;
import com.davidbracewell.apollo.optimization.CostGradientTuple;
import com.davidbracewell.apollo.optimization.Gradient;
import com.davidbracewell.apollo.optimization.WeightComponent;
import com.davidbracewell.apollo.optimization.loss.LossFunction;

/**
 * @author David B. Bracewell
 */
public class SequentialNetworkCostFunction implements CostFunction {
   private final SequentialNetwork network;
   private final LossFunction lossFunction;

   public SequentialNetworkCostFunction(SequentialNetwork network, LossFunction lossFunction) {
      this.network = network;
      this.lossFunction = lossFunction;
   }

   @Override
   public CostGradientTuple evaluate(Vector input, WeightComponent theta) {
      Vector y = input.getLabelVector(network.numberOfLabels());
      int numLayers = network.layers.length;

      Vector[] activations = new Vector[numLayers];
      for (int i = 0; i < numLayers; i++) {
         if (i == 0) {
            activations[i] = network.layers[i].forward(input);
         } else {
            activations[i] = network.layers[i].forward(activations[i - 1]);
         }
      }
      Vector predicted = activations[activations.length - 1];
      double totalError = lossFunction.loss(predicted, y);
      Vector[] deltas = new Vector[numLayers + 1];

      deltas[numLayers] = lossFunction.derivative(predicted, y).getBiasGradient();
      for (int i = numLayers - 1; i >= 0; i--) {
         deltas[i] = network.layers[i].backward(activations[i], deltas[i + 1]);
      }

      Gradient[] gradients = new Gradient[theta.size()];
      int index = 0;
      for (int i = 0; i < numLayers; i++) {
         Vector a = i == 0 ? input : activations[i - 1];
         if (network.layers[i].isOptimizable()) {
            gradients[index] = Gradient.of(deltas[i + 1].transpose().multiply(a.toMatrix()),
                                           deltas[i + 1]);
            index++;
         }
      }

      return CostGradientTuple.of(totalError, gradients);
   }

}//END OF SequentialNetworkCostFunction
