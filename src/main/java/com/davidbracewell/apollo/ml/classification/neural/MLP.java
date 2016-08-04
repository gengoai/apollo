package com.davidbracewell.apollo.ml.classification.neural;

import com.davidbracewell.apollo.linalg.DenseVector;
import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.ml.FeatureVector;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.classification.Classifier;
import com.davidbracewell.apollo.ml.classification.ClassifierLearner;
import com.davidbracewell.apollo.ml.data.Dataset;
import com.davidbracewell.logging.Logger;
import com.davidbracewell.tuple.Tuple2;
import com.google.common.base.Stopwatch;
import lombok.Getter;
import lombok.Setter;

import java.util.ArrayList;
import java.util.List;

/**
 * @author David B. Bracewell
 */
public class MLP extends ClassifierLearner {
  private static final Logger log = Logger.getLogger(MLP.class);
  private static final long serialVersionUID = 1L;
  private int[] hiddenLayers = new int[]{50};
  @Getter
  @Setter
  private double learningRate = 1;
  @Getter
  @Setter
  private double tolerance = 0.0001;
  @Getter
  @Setter
  private double maxIterations = 100;
  @Getter
  @Setter
  private boolean verbose = true;


  public int getHiddenLayerSize() {
    return hiddenLayers[0];
  }

  public void setHiddenLayerSize(int size) {
    hiddenLayers = new int[]{size};
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
        yVec.set(v.getLabel().intValue(), 1);
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
