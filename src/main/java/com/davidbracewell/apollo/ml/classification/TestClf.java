package com.davidbracewell.apollo.ml.classification;

import com.davidbracewell.apollo.linear.Axis;
import com.davidbracewell.apollo.linear.NDArray;
import com.davidbracewell.apollo.linear.NDArrayFactory;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.data.Dataset;
import com.davidbracewell.apollo.ml.optimization.activation.Activation;
import com.davidbracewell.apollo.ml.optimization.loss.CrossEntropyLoss;
import com.davidbracewell.guava.common.base.Stopwatch;
import lombok.val;

/**
 * @author David B. Bracewell
 */
public class TestClf extends ClassifierLearner {
   @Override
   protected void resetLearnerParameters() {

   }

   @Override
   protected Classifier trainImpl(Dataset<Instance> dataset) {
      LinearModel model = new LinearModel(this);
      val vectors = dataset.asVectors().collect();
      model.weights = NDArrayFactory.SPARSE_DOUBLE.zeros(model.numberOfLabels(), model.numberOfFeatures());
      model.bias = NDArrayFactory.SPARSE_DOUBLE.zeros(model.numberOfLabels());
      model.activation = Activation.SOFTMAX;
      double learningRate = 1;

      NDArray[] weights = new NDArray[model.numberOfLabels()];
      for (int i = 0; i < weights.length; i++) {
         weights[i] = NDArrayFactory.SPARSE_DOUBLE.zeros(model.numberOfFeatures());
      }

      CrossEntropyLoss loss = new CrossEntropyLoss();
      for (int iteration = 0; iteration < 100; iteration++) {
         val timer = Stopwatch.createStarted();
         double lr = learningRate / (1.0 + 0.01 * iteration);
         for (NDArray vector : vectors) {
            val yHat = model.activation.apply(model.weights.dot(vector.T(), Axis.ROW).T().add(model.bias));
            val grad = loss.derivative(yHat, vector.getLabelAsNDArray(model.numberOfLabels()));
            grad.forEach(entry -> {
               val xv = vector.mul(entry.getValue() * lr).T();
               model.weights.subiVector(entry.getIndex(), xv, Axis.ROW);
               model.bias.decrement(entry.getIndex(), entry.getValue() * lr);
            });
         }
         System.out.println(timer);
      }
      return model;
   }
}// END OF TestClf
