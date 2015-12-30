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
import com.davidbracewell.function.Unchecked;
import com.davidbracewell.io.Resources;
import com.davidbracewell.io.resource.Resource;
import com.davidbracewell.stream.Streams;
import com.davidbracewell.string.StringUtils;
import com.google.common.base.Joiner;

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.stream.Collectors;

/**
 * @author David B. Bracewell
 */
public class MLP extends ClassifierLearner {
  private static final long serialVersionUID = 1L;
  private int[] hiddenLayers = new int[]{50};
  private double learningRate = 0.3;
  private double momentum = 1.0;
  private double maxIterations = 100;

  public static void main(String[] args) throws Exception {
//    Dataset<Instance> dataset = Dataset.classification()
//      .localSource(
//        Arrays.asList(
//          Instance.create(Arrays.asList(Feature.TRUE("3")), "false"),
//          Instance.create(Arrays.asList(Feature.TRUE("2"), Feature.TRUE("3")), "true"),
//          Instance.create(Arrays.asList(Feature.TRUE("1"), Feature.TRUE("3")), "true"),
//          Instance.create(Arrays.asList(Feature.TRUE("1"), Feature.TRUE("2"), Feature.TRUE("3")), "false")
//        ).stream()
//      )
//      .build();

    Dataset<Instance> dataset = Dataset.classification()
      .localSource(
        Resources.fromFile("/home/david/Downloads/Data/SomasundaranWiebe-politicalDebates/abortion")
          .getChildren()
          .stream()
          .map(Unchecked.function(Resource::readToString))
          .map(doc -> {
            String[] parts = doc.split("\n+");
            String label = parts[0].split("=")[1];
            String content = Joiner.on('\n').join(Arrays.copyOfRange(parts, 3, parts.length)).toLowerCase();
            Map<String, Long> counts = Streams.of(content.split("[^A-Za-z]+")).filter(s -> !StringUtils.isNullOrBlank(s)).countByValue();
            List<Feature> features = counts.entrySet().stream().map(e -> Feature.real(e.getKey(), e.getValue())).collect(Collectors.toList());
            return Instance.create(features, label);
          })
      ).build()
      .shuffle(new Random(123));

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
      Activation af = i +1 == model.layers.length ? new Sigmoid() : new TanH();
      model.layers[i] = new Layer(inputSize, outputSize, af);
    }

    int nL = model.layers.length;
    int outputLayer = nL - 1;

    Layer[] net = model.layers;
    for (int iteration = 0; iteration < maxIterations; iteration++) {
      double error = 0;
      for (Instance instance : dataset) {
        FeatureVector v = instance.toVector(model.getEncoderPair());
        Vector yVec = new DenseVector(model.numberOfLabels());
        yVec.set((int) v.getLabel(), 1);

        Matrix[] outputs = new Matrix[nL];
        Matrix input = v.toMatrix();
        for (int i = 0; i < nL; i++) {
          outputs[i] = model.layers[i].evaluate(input);
          input = outputs[i];
        }

        Matrix[] deltas = new Matrix[nL];
        for (int i = 0; i < model.numberOfLabels(); i++) {
          error += (yVec.get(i) - input.get(0, i));
        }
        deltas[outputLayer] = yVec.toMatrix().subtract(input).scaleSelf(net[outputLayer].gradient(input));
        for (int i = nL - 2; i >= 0; i--) {
          deltas[i] = deltas[i + 1].multiply(net[i + 1].getMatrix().transpose()).scaleSelf(net[i].gradient(outputs[i])).scale(learningRate);
        }

        net[0].getMatrix().addSelf(v.transpose().multiply(deltas[0]));
        for (int i = 1; i < nL; i++) {
          net[i].getMatrix().addSelf(outputs[i - 1].transpose().multiply(deltas[i]));
        }
      }
      System.err.println(iteration + " :  " + error);
    }

    return model;
  }

  @Override
  public void reset() {

  }

}// END OF MLP
