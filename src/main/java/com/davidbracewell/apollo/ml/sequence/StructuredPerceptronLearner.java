package com.davidbracewell.apollo.ml.sequence;

import com.davidbracewell.apollo.linalg.DenseVector;
import com.davidbracewell.apollo.linalg.DynamicSparseVector;
import com.davidbracewell.apollo.linalg.SparseVector;
import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.ml.Dataset;
import com.davidbracewell.apollo.ml.Encoder;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.logging.Logger;
import com.google.common.base.Stopwatch;
import com.google.common.collect.HashBasedTable;
import com.google.common.collect.Table;

import java.text.DecimalFormat;

/**
 * The type Structured perceptron learner.
 *
 * @author David B. Bracewell
 */
public class StructuredPerceptronLearner extends SequenceLabelerLearner {
  private static final Logger log = Logger.getLogger(StructuredPerceptronLearner.class);
  private final int VALUE = 0;
  private final int ITER = 1;
  private final int EVENT = 2;
  private TransitionFeatures transitionFeatures = TransitionFeatures.FIRST_ORDER;
  private int maxIterations = 10;
  private double learningRate = 1;
  private double tolerance = 0.00001;
  private Vector[] averages;
  private Table<Integer, Integer, int[]> updates;

  @Override
  protected SequenceLabeler trainImpl(Dataset<Sequence> dataset) {
    StructuredPerceptron perception = new StructuredPerceptron(
      dataset.getLabelEncoder(),
      dataset.getFeatureEncoder(),
      dataset.getPreprocessors(),
      transitionFeatures
    );
    perception.setDecoder(getDecoder());


    final Encoder lblEncoder = dataset.getLabelEncoder();
    dataset.encode();
    int nC = perception.numberOfLabels();
    perception.weights = new SparseVector[nC];

    averages = new DynamicSparseVector[nC];
    updates = HashBasedTable.create();

    perception.biases = new DenseVector(nC);
    for (int i = 0; i < nC; i++) {
      perception.weights[i] = new DynamicSparseVector(perception::numberOfFeatures);
      averages[i] = new DynamicSparseVector(perception::numberOfFeatures);
    }


    double oldOldError = 0;
    double oldError = 0;
    final DecimalFormat formatter = new DecimalFormat("###.00%");

    for (int itr = 0; itr < maxIterations; itr++) {
      Stopwatch sw = Stopwatch.createStarted();
      double count = 0;
      double correct = 0;
      int c = 0;

      for (Sequence sequence : dataset) {

        //Label the sequence and check for errors
        LabelingResult result = perception.label(sequence);
        boolean update = false;

        for (int i = 0; i < sequence.size(); i++) {
          count++;
          if (!sequence.get(i).getLabel().equals(result.getLabel(i))) {
            update = true;
          } else {
            correct++;
          }
        }

        if (update) {
          for (ContextualIterator<Instance> iterator = sequence.iterator(); iterator.hasNext(); ) {
            Instance current = iterator.next();
            int y = (int) lblEncoder.encode(current.getLabel());
            int yHat = (int) lblEncoder.encode(result.getLabel(iterator.getIndex()));

            boolean changed = y != yHat;
            for (int j = 1; !changed && iterator.getIndex() - j >= 0 && j <= transitionFeatures.getHistorySize(); j++) {
              if (!iterator.getPrevious(j).getLabel().equals(result.getLabel(iterator.getIndex() - j))) {
                changed = true;
              }
            }

            if (changed) {
              perception.biases.increment(y);
              perception.biases.decrement(yHat);
              current.forEach(feature -> {
                perception.weights[y].increment((int) perception.getFeatureEncoder().encode(feature.getName()), learningRate);
                perception.weights[yHat].decrement((int) perception.getFeatureEncoder().encode(feature.getName()), learningRate);
              });
              transitionFeatures.extract(iterator)
                .forEach(feature -> perception.weights[y].increment((int) perception.getFeatureEncoder().encode(feature.getName()), learningRate));
              transitionFeatures.extract(result, iterator.getIndex())
                .forEach(feature -> perception.weights[yHat].decrement((int) perception.getFeatureEncoder().encode(feature.getName()), learningRate));
            }
          }

        }
        c++;
      }


      sw.stop();
      log.info("iteration={0} accuracy={1} ({2}/{3}) [completed in {4}]", itr + 1, formatter.format(correct / count), correct, count, sw);
      if (count - correct == 0) {
        break;
      }
      double error = count - correct;
      error /= count;
      if (itr > 2) {
        if (Math.abs(error - oldError) < tolerance && Math.abs(error - oldOldError) < tolerance) {
          break;
        }
      }
      oldOldError = oldError;
      oldError = error;
      dataset.shuffle();
    }


    return perception;
  }

  @Override
  public void reset() {
    averages = null;
    updates = null;
  }


  /**
   * Gets max iterations.
   *
   * @return the max iterations
   */

  public int getMaxIterations() {
    return maxIterations;
  }

  /**
   * Sets max iterations.
   *
   * @param maxIterations the max iterations
   */
  public void setMaxIterations(int maxIterations) {
    this.maxIterations = maxIterations;
  }

  /**
   * Gets learning rate.
   *
   * @return the learning rate
   */
  public double getLearningRate() {
    return learningRate;
  }

  /**
   * Sets learning rate.
   *
   * @param learningRate the learning rate
   */
  public void setLearningRate(double learningRate) {
    this.learningRate = learningRate;
  }

  /**
   * Gets tolerance.
   *
   * @return the tolerance
   */
  public double getTolerance() {
    return tolerance;
  }

  /**
   * Sets tolerance.
   *
   * @param tolerance the tolerance
   */
  public void setTolerance(double tolerance) {
    this.tolerance = tolerance;
  }

  /**
   * Gets transition features.
   *
   * @return the transition features
   */
  public TransitionFeatures getTransitionFeatures() {
    return transitionFeatures;
  }

  /**
   * Sets transition features.
   *
   * @param transitionFeatures the transition features
   */
  public void setTransitionFeatures(TransitionFeatures transitionFeatures) {
    this.transitionFeatures = transitionFeatures;
  }

}// END OF StructuredPerceptronLearner
