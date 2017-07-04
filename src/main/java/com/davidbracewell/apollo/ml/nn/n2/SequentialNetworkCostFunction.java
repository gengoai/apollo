package com.davidbracewell.apollo.ml.nn.n2;

import com.davidbracewell.apollo.linalg.Matrix;
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
   public CostGradientTuple evaluate(Iterable<Vector> vectors, WeightComponent theta) {
      for (Vector input : vectors) {

      }

      return null;
   }

   @Override
   public CostGradientTuple evaluate(Vector input, WeightComponent theta) {
      Vector y = input.getLabelVector(network.numberOfLabels());
      Vector[] activations = new Vector[network.layers.length];
      for (int i = 0; i < network.layers.length; i++) {
         if (i == 0) {
            activations[i] = network.layers[i].forward(input);
         } else {
            activations[i] = network.layers[i].forward(activations[i - 1]);
         }
      }
      Vector predicted = activations[activations.length - 1];
      double totalError = lossFunction.loss(predicted, y);
      Gradient[] deltas = new Gradient[network.layers.length];
      deltas[network.layers.length - 1] = lossFunction.derivative(predicted, y);
      for (int i = network.layers.length - 1; i >= 0; i--) {
         Vector a = i == 0 ? input : activations[i - 1];
         deltas[i] = network.layers[i].backward(a, activations[i], deltas[i + 1], null, 0);
      }

      Matrix[] gradients = new Matrix[theta.size()];
      int index = 0;
      for (int i = 0; i < network.layers.length; i++) {
         Vector a = i == 0 ? input : activations[i - 1];
         if (network.layers[i].isOptimizable()) {
            gradients[index] = deltas[i + 1].transpose().multiply(a.toMatrix());
            index++;
         }
      }

      return null;
   }

}//END OF SequentialNetworkCostFunction
