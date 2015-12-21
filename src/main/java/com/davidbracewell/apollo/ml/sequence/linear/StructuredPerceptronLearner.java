package com.davidbracewell.apollo.ml.sequence.linear;

import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.ml.Dataset;
import com.davidbracewell.apollo.ml.Feature;
import com.davidbracewell.apollo.ml.FeatureVector;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.sequence.*;
import com.davidbracewell.collection.Collect;
import com.davidbracewell.logging.Logger;
import com.google.common.base.Stopwatch;
import com.google.common.collect.Lists;

import java.text.DecimalFormat;
import java.util.List;

/**
 * The type Structured perceptron learner.
 *
 * @author David B. Bracewell
 */
public class StructuredPerceptronLearner extends SequenceLabelerLearner {
  private static final Logger log = Logger.getLogger(StructuredPerceptronLearner.class);
  private int maxIterations = 10;
  private double tolerance = 0.00001;
  private Vector[] cWeights;


  @Override
  protected SequenceLabeler trainImpl(Dataset<Sequence> dataset) {

    StructuredPerceptron model = new StructuredPerceptron(
      dataset.getLabelEncoder(),
      dataset.getFeatureEncoder(),
      dataset.getPreprocessors(),
      transitionFeatures
    );
    model.setDecoder(getDecoder());

    int nC = model.numberOfLabels();

    model.weights = new Vector[nC];
    cWeights = new Vector[nC];
    for (int i = 0; i < nC; i++) {
      model.weights[i] = new FeatureVector(model.getFeatureEncoder());
      cWeights[i] = new FeatureVector(model.getFeatureEncoder());
    }


    double oldOldError = 0;
    double oldError = 0;
    final DecimalFormat formatter = new DecimalFormat("###.00%");

    List<Sequence> sequenceList = Lists.newLinkedList(Collect.asIterable(dataset.iterator()));
    int c = 1;
    for (int itr = 0; itr < maxIterations; itr++) {
      Stopwatch sw = Stopwatch.createStarted();

      double count = 0;
      double correct = 0;

      for (Sequence sequence : sequenceList) {

        LabelingResult lblResult = model.label(sequence);

        double diff = 0;
        for (ContextualIterator<Instance> iterator = sequence.iterator(); iterator.hasNext(); ) {
          count++;
          if (!iterator.next().getLabel().equals(lblResult.getLabel(iterator.getIndex()))) {
            diff++;
          } else {
            correct++;
          }
        }


        if (diff > 0) {
          for (ContextualIterator<Instance> iterator = sequence.iterator(); iterator.hasNext(); ) {
            Instance instance = iterator.next();
            int y = (int) model.getLabelEncoder().encode(instance.getLabel());
            int yHat = (int) model.getLabelEncoder().encode(lblResult.getLabel(iterator.getIndex()));
            if (y != yHat) {
              for (Feature feature : instance) {
                int fid = (int) model.getFeatureEncoder().encode(feature.getName());
                model.weights[yHat].decrement(fid);
                model.weights[y].increment(fid);
                cWeights[yHat].decrement(fid);
                cWeights[y].increment(fid);
              }
              for (String feature : Collect.asIterable(transitionFeatures.extract(lblResult, iterator.getIndex()))) {
                int fid = (int) model.getFeatureEncoder().encode(feature);
                model.weights[yHat].decrement(fid);
                cWeights[yHat].decrement(fid);
              }
              for (String feature : Collect.asIterable(transitionFeatures.extract(iterator))) {
                int fid = (int) model.getFeatureEncoder().encode(feature);
                model.weights[y].increment(fid);
                cWeights[y].increment(fid);
              }
            }
          }

          c++;
        }

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

//      Collections.shuffle(sequenceList);
//      dataset.shuffle();
    }

    final double C = c;
    for (int ci = 0; ci < nC; ci++) {
      Vector v = model.weights[ci];
      cWeights[ci].forEachSparse(entry -> v.decrement(entry.index, entry.value / C));
    }

    return model;
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


}// END OF StructuredPerceptronLearner
