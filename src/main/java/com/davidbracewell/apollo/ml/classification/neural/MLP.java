package com.davidbracewell.apollo.ml.classification.neural;

import com.davidbracewell.apollo.DifferentiableFunction;
import com.davidbracewell.apollo.linalg.DenseVector;
import com.davidbracewell.apollo.linalg.SparseVector;
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
import java.util.Collections;
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

  @Override
  protected Classifier trainImpl(Dataset<Instance> dataset) {
    NeuralNetwork model = new NeuralNetwork(
      dataset.getEncoderPair(),
      dataset.getPreprocessors()
    );

    model.layers = new Layer[hiddenLayers.length + 2];
    int inputsize = model.numberOfFeatures();
    model.layers[0] = new Layer(inputsize, inputsize, new Linear());
    for (int i = 1; i < model.layers.length; i++) {
      int hlI = i - 1;
      int outputSize = hlI == hiddenLayers.length ? model.numberOfLabels() : hiddenLayers[hlI];
      DifferentiableFunction af = (i + 1 == model.layers.length) ? new Sigmoid() : new TanH();
      model.layers[i] = new Layer(inputsize, outputSize, af);
      if (hlI < hiddenLayers.length) {
        inputsize = hiddenLayers[hlI];
      }
    }

    int nL = model.layers.length;
    int outputLayer = nL - 1;

    for (int iteration = 0; iteration < maxIterations; iteration++) {
      for (Instance instance : dataset) {
        FeatureVector v = instance.toVector(model.getEncoderPair());
        Vector yVec = new DenseVector(model.numberOfLabels());
        yVec.set((int) v.getLabel(), 1);

        List<Vector> inputs = new ArrayList<>();
        List<Vector> gradients = new ArrayList<>();
        Vector input = v;
        for (int i = 0; i < nL; i++) {
          inputs.add(model.layers[i].evaluate(input));
          gradients.add(model.layers[i].gradient(input));
          input = inputs.get(i);
          System.out.println(input);
        }

        System.err.println(inputs);
        System.err.println(gradients);

        Vector delta = yVec.subtract(inputs.get(outputLayer)).multiplySelf(gradients.get(outputLayer));
        List<Vector> deltas = new ArrayList<>();
        deltas.add(delta);

        for (int i = nL - 2; i >= 1; i--) {
          delta = model.layers[i + 1].multiply(delta).multiplySelf(gradients.get(i));
          deltas.add(delta);
        }
        deltas.add(new SparseVector(model.numberOfFeatures()));
        Collections.reverse(deltas);

        model.layers[1].update(v.multiply(deltas.get(0)));
        for (int i = 2; i < nL; i++) {
          model.layers[i].update(inputs.get(i).multiply(deltas.get(i - 1)));
        }


      }
    }

    return model;
  }

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
    mlp.hiddenLayers = new int[]{3};

    Classifier c = mlp.train(dataset);

    ClassifierEvaluation evaluation = new ClassifierEvaluation();
    evaluation.evaluate(c, dataset);
    evaluation.output(System.out);


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
