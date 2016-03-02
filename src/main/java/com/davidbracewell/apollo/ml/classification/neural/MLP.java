package com.davidbracewell.apollo.ml.classification.neural;

import com.davidbracewell.apollo.linalg.DenseVector;
import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.ml.Dataset;
import com.davidbracewell.apollo.ml.Feature;
import com.davidbracewell.apollo.ml.FeatureVector;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.classification.Classifier;
import com.davidbracewell.apollo.ml.classification.ClassifierEvaluation;
import com.davidbracewell.apollo.ml.classification.ClassifierLearner;
import com.davidbracewell.apollo.ml.preprocess.PreprocessorList;
import com.davidbracewell.apollo.ml.preprocess.filter.CountFilter;
import com.davidbracewell.function.Unchecked;
import com.davidbracewell.io.Resources;
import com.davidbracewell.io.resource.Resource;
import com.davidbracewell.logging.Logger;
import com.davidbracewell.stream.Streams;
import com.davidbracewell.string.StringUtils;
import com.davidbracewell.tuple.Tuple2;
import com.google.common.base.Joiner;
import com.google.common.base.Stopwatch;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.stream.Collectors;

/**
 * @author David B. Bracewell
 */
public class MLP extends ClassifierLearner {
  private static final Logger log = Logger.getLogger(MLP.class);
  private static final long serialVersionUID = 1L;
  private int[] hiddenLayers = new int[]{50};
  private double learningRate = 1;
  private double tolerance = 0.0001;
  private double maxIterations = 100;
  private boolean verbose = true;

  public static void main(String[] args) throws Exception {
    Dataset<Instance> dataset = Dataset.classification()
      .localSource(
        Resources.fromFile("/data/Downloads/SomasundaranWiebe-politicalDebates/abortion")
          .getChildren()
          .stream()
          .map(Unchecked.function(Resource::readToString))
          .map(doc -> {
            String[] parts = doc.split("\n+");
            String label = parts[0].split("=")[1];
            String content = Joiner.on('\n').join(Arrays.copyOfRange(parts, 3, parts.length)).toLowerCase();
            Map<String, Long> counts = Streams.of(content.split("[^A-Za-z]+"))
              .filter(s -> !StringUtils.isNullOrBlank(s)).map(String::toLowerCase).countByValue();
            List<Feature> features = counts.entrySet().stream().map(e -> Feature.real(e.getKey(), e.getValue())).collect(Collectors.toList());
            return Instance.create(features, label);
          })
      ).build()
      .shuffle(new Random(123));

    MLP mlp = new MLP();
    mlp.learningRate=1.0;
    mlp.maxIterations=500;
    mlp.hiddenLayers = new int[]{4};

    dataset.split(0.8).forEach((train, test) -> {
      Classifier c = mlp.train(train.preprocess(PreprocessorList.create(new CountFilter(d -> d >= 5))));
      ClassifierEvaluation evaluation = new ClassifierEvaluation();
      evaluation.evaluate(c, test);
      evaluation.output(System.out);

    });


  }

  @Override
  protected Classifier trainImpl(Dataset<Instance> dataset) {
    NeuralNetwork model = new NeuralNetwork(
      dataset.getEncoderPair(),
      dataset.getPreprocessors()
    );

    List<Tuple2<Integer, Activation>> layers = new ArrayList<>();
    for (int hiddenLayer : hiddenLayers) {
      layers.add(Tuple2.of(hiddenLayer, new Sigmoid()));
    }
    model.network = new Network(layers, dataset.getEncoderPair());

    double lastError = 0;
    double lastLastError = 0;
    Stopwatch sw = Stopwatch.createUnstarted();
    for (int iteration = 0; iteration < maxIterations; iteration++) {
      double error = 0;
      double N = 0;
      sw.start();
      for (Instance instance : dataset) {
        N++;
        FeatureVector v = instance.toVector(model.getEncoderPair());
        Vector yVec = new DenseVector(model.numberOfLabels());
          yVec.set((int) v.getLabel(), 1);
        error += model.network.backprop(
          model.network.forward(v),
          v,
          yVec,
          learningRate
        );
      }
      error = error / N;
      sw.stop();
      if (verbose) {
        log.info("Iteration {0}: total error={1,number,0.00000} [{2}]", (iteration + 1), error, sw);
      }
      sw.reset();

      if (error == 0 ||
        (iteration > 2 &&
          Math.signum(error) == Math.signum(lastError) && Math.abs(error - lastError) < tolerance &&
          Math.signum(lastError) == Math.signum(lastLastError) && Math.abs(lastError - lastLastError) < tolerance)) {
        break;
      }
      lastLastError = lastError;
      lastError = error;

    }

    return model;
  }

  @Override
  public void reset() {

  }

}// END OF MLP
