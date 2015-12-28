package com.davidbracewell.apollo.ml.classification.neural;

import com.davidbracewell.apollo.DifferentiableFunction;
import com.davidbracewell.apollo.Sigmoid;
import com.davidbracewell.apollo.TanH;
import com.davidbracewell.apollo.linalg.DenseVector;
import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.ml.Dataset;
import com.davidbracewell.apollo.ml.FeatureVector;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.classification.Classification;
import com.davidbracewell.apollo.ml.classification.Classifier;
import com.davidbracewell.apollo.ml.classification.ClassifierLearner;

/**
 * @author David B. Bracewell
 */
public class MLP extends ClassifierLearner {

  private int[] hiddenLayers = new int[]{50};
  private double learningRate = 1.0;

  @Override
  protected Classifier trainImpl(Dataset<Instance> dataset) {
    NeuralNetwork model = new NeuralNetwork(
      dataset.getEncoderPair(),
      dataset.getPreprocessors()
    );

    model.layers = new Layer[hiddenLayers.length + 1];
    int inputsize = model.getEncoderPair().numberOfFeatures() + 1;
    for (int i = 0; i < model.layers.length; i++) {
      DifferentiableFunction af = (i + 1 == model.layers.length) ? new Sigmoid() : new TanH();
      model.layers[i] = new Layer(inputsize, hiddenLayers[i] + 1, af);
      inputsize = hiddenLayers[i] + 1;
    }

    for (Instance instance : dataset) {
      FeatureVector v = instance.toVector(model.getEncoderPair());
      Classification result = model.classify(v);
      double y = v.getLabel();
      double yHat = result.getEncodedResult();
      if (y != yHat) {
        double[] predicted = result.distribution();
        double e = 0;
        for (int i = 0; i < predicted.length; i++) {
          e += Math.pow(predicted[i] - (y == i ? 1.0 : 0.0), 2);
        }
        e /= 2.0;

      }
    }

    return model;
  }

  private void graidents(double[] predicted, NeuralNetwork model) {
    Vector input = DenseVector.wrap(predicted);
    for (int c = model.layers.length - 1; c >= 0; c--) {
      model.layers[c].gradient(input);
    }
  }

  @Override
  public void reset() {

  }

}// END OF MLP
