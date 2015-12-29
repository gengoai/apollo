package com.davidbracewell.apollo.ml.classification.neural;

import com.davidbracewell.apollo.DifferentiableFunction;
import com.davidbracewell.apollo.linalg.DenseVector;
import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.ml.Dataset;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.classification.Classifier;
import com.davidbracewell.apollo.ml.classification.ClassifierLearner;

/**
 * @author David B. Bracewell
 */
public class MLP extends ClassifierLearner {
  private static final long serialVersionUID = 1L;
  private int[] hiddenLayers = new int[]{50};
  private double learningRate = 1.0;
  private double momentum = 1.0;

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
