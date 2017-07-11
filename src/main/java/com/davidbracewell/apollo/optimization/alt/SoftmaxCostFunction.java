package com.davidbracewell.apollo.optimization.alt;

import com.davidbracewell.apollo.linalg.DenseVector;
import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.optimization.activation.Activation;

/**
 * @author David B. Bracewell
 */
public class SoftmaxCostFunction implements CostFunction {
   private final int numberOfLabels;
   private LossFunction lossFunction = new LogLoss();

   public SoftmaxCostFunction(int numberOfLabels) {
      this.numberOfLabels = numberOfLabels;
   }

   @Override
   public CostGradientTuple evaluate(Vector vector, WeightVector theta) {
      Vector predicted = new DenseVector(numberOfLabels);
      int l = 0;
      for (int offset = 0; offset < theta.getWeights().dimension(); offset += vector.dimension()) {
         for (int i = offset; i < vector.dimension(); i++) {
            predicted.increment(l, vector.get(i - offset) * theta.getWeights().get(offset));
         }
         l++;
      }
      predicted = Activation.SOFTMAX.apply(predicted);
      Vector y = vector.getLabelVector(numberOfLabels);

      double loss = lossFunction.loss(predicted, y);
      Vector gradient = gradient(predicted, y);

      return CostGradientTuple.of(loss, Gradient.of(gradient, gradient.sum()));
   }

   public Vector gradient(Vector predicted, Vector y) {
      Vector gradient = Vector.dZeros(predicted.dimension());
      for (int i = 0; i < predicted.dimension(); i++) {
         double vi = predicted.get(i);
         double sum = 0;
         for (int j = 0; j < y.dimension(); j++) {
            if (i == j) {
               sum += vi * (1 - vi);
            } else {
               sum += -vi * y.get(j);
            }
         }
         gradient.set(i, sum);
      }
      return gradient;
   }

}// END OF SoftmaxCostFunction
