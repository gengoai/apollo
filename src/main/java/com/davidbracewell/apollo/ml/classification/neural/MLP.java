package com.davidbracewell.apollo.ml.classification.neural;

import com.davidbracewell.apollo.linalg.DenseVector;
import com.davidbracewell.apollo.linalg.Matrix;
import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.ml.Dataset;
import com.davidbracewell.apollo.ml.Feature;
import com.davidbracewell.apollo.ml.FeatureVector;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.classification.Classifier;
import com.davidbracewell.apollo.ml.classification.ClassifierEvaluation;
import com.davidbracewell.apollo.ml.classification.ClassifierLearner;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * @author David B. Bracewell
 */
public class MLP extends ClassifierLearner {
  private static final long serialVersionUID = 1L;
  private int[] hiddenLayers = new int[]{50};
  private double learningRate = 1.0;
  private double momentum = 1.0;
  private double maxIterations = 60000;

  public static void main(String[] args) throws Exception {
    Dataset<Instance> dataset = Dataset.classification()
      .localSource(
        Arrays.asList(
          Instance.create(Arrays.asList(Feature.TRUE("3")), "false"),
          Instance.create(Arrays.asList(Feature.TRUE("2"), Feature.TRUE("3")), "true"),
          Instance.create(Arrays.asList(Feature.TRUE("1"), Feature.TRUE("3")), "true"),
          Instance.create(Arrays.asList(Feature.TRUE("1"), Feature.TRUE("2"), Feature.TRUE("3")), "false")
        ).stream()
      )
      .build();

    MLP mlp = new MLP();
    mlp.hiddenLayers = new int[]{4};

    Classifier c = mlp.train(dataset);

    ClassifierEvaluation evaluation = new ClassifierEvaluation();
    evaluation.evaluate(c, dataset);
    evaluation.output(System.out);


  }

  @Override
  protected Classifier trainImpl(Dataset<Instance> dataset) {
    NeuralNetwork model = new NeuralNetwork(
      dataset.getEncoderPair(),
      dataset.getPreprocessors()
    );

    model.layers = new Layer[hiddenLayers.length + 1];
    for (int i = 0; i < model.layers.length; i++) {
      int inputSize = (i == 0 ? model.numberOfFeatures() : model.layers[i - 1].getMatrix().numberOfColumns());
      int outputSize = (i == hiddenLayers.length ? model.numberOfLabels() : hiddenLayers[i]);
      Activation af = (i + 1 == model.layers.length) ? new Sigmoid() : new TanH();
      model.layers[i] = new Layer(inputSize, outputSize, af);
    }

    int nL = model.layers.length;
    int outputLayer = nL - 1;

    for (int iteration = 0; iteration < maxIterations; iteration++) {
      for (Instance instance : dataset) {
        FeatureVector v = instance.toVector(model.getEncoderPair());
        Vector yVec = new DenseVector(model.numberOfLabels());
        yVec.set((int) v.getLabel(), 1);

        List<Matrix> inputs = new ArrayList<>();
        Matrix input = v.toMatrix();
        for (int i = 0; i < nL; i++) {
          inputs.add(model.layers[i].evaluate(input));
          input = inputs.get(i);
        }

        Matrix delta = yVec.toMatrix().subtract(input).scaleSelf(model.layers[outputLayer].gradient(input));
        Matrix[] deltas = new Matrix[model.layers.length];
        deltas[outputLayer] = delta;
        for (int i = nL - 2; i >= 0; i--) {
          delta = delta.multiply(model.layers[i + 1].getMatrix().transpose()).scaleSelf(model.layers[i].gradient(inputs.get(i)));
          deltas[i] = delta;
        }

        model.layers[0].getMatrix().addSelf(v.transpose().multiply(deltas[0]));
        System.out.println(model.layers[0].getMatrix());
        for (int i = 1; i < nL; i++) {
          model.layers[i].getMatrix().addSelf(inputs.get(i - 1).transpose().multiply(deltas[i]));
        }
      }
    }

    return model;
  }

  @Override
  public void reset() {

  }

}// END OF MLP
