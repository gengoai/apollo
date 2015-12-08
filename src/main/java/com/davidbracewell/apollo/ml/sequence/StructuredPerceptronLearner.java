package com.davidbracewell.apollo.ml.sequence;

import com.davidbracewell.apollo.linalg.DynamicSparseVector;
import com.davidbracewell.apollo.linalg.SparseVector;
import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.ml.Dataset;
import com.davidbracewell.apollo.ml.Encoder;
import com.davidbracewell.apollo.ml.Feature;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.collection.Collect;
import com.davidbracewell.logging.Logger;
import com.google.common.base.Stopwatch;

import java.text.DecimalFormat;

/**
 * The type Structured perceptron learner.
 *
 * @author David B. Bracewell
 */
public class StructuredPerceptronLearner extends SequenceLabelerLearner {
  private static final Logger log = Logger.getLogger(StructuredPerceptronLearner.class);
  private TransitionFeatures transitionFeatures = TransitionFeatures.FIRST_ORDER;
  private int maxIterations = 10;
  private double tolerance = 0.00001;
  private Vector[] cWeights;

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

    int nC = perception.numberOfLabels();
    perception.weights = new SparseVector[nC];

    cWeights = new DynamicSparseVector[nC];
    for (int i = 0; i < nC; i++) {
      perception.weights[i] = new DynamicSparseVector(perception::numberOfFeatures);
      cWeights[i] = new DynamicSparseVector(perception::numberOfFeatures);
    }


    double oldOldError = 0;
    double oldError = 0;
    final DecimalFormat formatter = new DecimalFormat("###.00%");

    int c = 1;
    Vector[] counts = new Vector[nC];
    for (int i = 0; i < nC; i++) {
      counts[i] = new DynamicSparseVector(perception.getFeatureEncoder()::size);
    }
    for (int itr = 0; itr < maxIterations; itr++) {
      Stopwatch sw = Stopwatch.createStarted();
      double count = 0;
      double correct = 0;

      for (Sequence sequence : dataset) {
        for (int i = 0; i < nC; i++) {
          counts[i].zero();
        }

        LabelingResult lblResult = perception.label(sequence);
        for (ContextualIterator<Instance> iterator = sequence.iterator(); iterator.hasNext(); ) {
          count++;
          Instance instance = iterator.next();
          int y = (int) perception.getLabelEncoder().encode(instance.getLabel());
          int yHat = (int) perception.getLabelEncoder().encode(lblResult.getLabel(iterator.getIndex()));
          if (y != yHat) {
            for (Feature feature : instance) {
              int fid = (int) perception.getFeatureEncoder().encode(feature.getName());
              counts[yHat].decrement(fid);
              counts[y].increment(fid);
            }
            for (Feature feature : transitionFeatures.extract(lblResult, iterator.getIndex())) {
              int fid = (int) perception.getFeatureEncoder().encode(feature.getName());
              counts[yHat].decrement(fid);
            }
            for (Feature feature : transitionFeatures.extract(iterator)) {
              int fid = (int) perception.getFeatureEncoder().encode(feature.getName());
              counts[y].increment(fid);
            }
          } else {
            correct++;
          }
        }

        for (int i = 0; i < nC; i++) {
          for (Vector.Entry entry : Collect.asIterable(counts[i].nonZeroIterator())) {
            cWeights[i].increment(entry.index, c);
            perception.weights[i].increment(entry.index, entry.getValue());
          }
        }
        c++;

      }


      sw.stop();
      log.info("iteration={0} accuracy={1} ({2}/{3}) [completed in {4}]", itr + 1, formatter.format(correct / count), correct, count, sw);

      if (count - correct == 0) {
        break;
      }

      double error = (count - correct) / count;
      if (itr > 2 && Math.abs(error - oldError) < tolerance && Math.abs(error - oldOldError) < tolerance) {
        break;
      }

      oldOldError = oldError;
      oldError = error;
      dataset.shuffle();
    }

    final double C = c;
    for (int ci = 0; ci < nC; ci++) {
      Vector v = perception.weights[ci];
      cWeights[ci].forEachSparse(entry -> v.decrement(entry.index, entry.value / C));
    }

    return perception;
  }

  @Override
  public void reset() {
    cWeights = null;
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
