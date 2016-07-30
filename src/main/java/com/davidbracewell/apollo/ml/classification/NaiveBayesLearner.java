package com.davidbracewell.apollo.ml.classification;

import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.data.Dataset;
import com.davidbracewell.collection.Collect;
import lombok.Getter;
import lombok.NonNull;
import lombok.Setter;

import java.util.Arrays;

/**
 * @author David B. Bracewell
 */
public class NaiveBayesLearner extends ClassifierLearner {
  private static final long serialVersionUID = 1L;
  @Getter
  @Setter(onParam = @_({@NonNull}))
  private volatile NaiveBayes.ModelType modelType;

  public NaiveBayesLearner() {
    this(NaiveBayes.ModelType.Bernoulli);
  }

  public NaiveBayesLearner(@NonNull NaiveBayes.ModelType modelType) {
    this.modelType = modelType;
  }

  @Override
  protected NaiveBayes trainImpl(Dataset<Instance> dataset) {
    NaiveBayes model = new NaiveBayes(dataset.getEncoderPair(), dataset.getPreprocessors(), modelType);
    model.conditionals = new double[model.numberOfFeatures()][model.numberOfLabels()];
    model.priors = new double[model.numberOfLabels()];
    double[] labelCounts = new double[model.numberOfLabels()];

    double N = 0;
    for (Instance instance : dataset) {
      if (instance.hasLabel()) {
        N++;
        int ci = (int) model.encodeLabel(instance.getLabel());
        model.priors[ci] += instance.getWeight();
        Vector vector = instance.toVector(dataset.getEncoderPair());
        for (Vector.Entry entry : Collect.asIterable(vector.nonZeroIterator())) {
          labelCounts[ci] += entry.value;
          model.conditionals[entry.getIndex()][ci] += instance.getWeight() * modelType.convertValue(entry.value);
        }
      }
    }

    double V = model.numberOfFeatures();
    for (int featureIndex = 0; featureIndex < model.conditionals.length; featureIndex++) {
      double[] tmp = Arrays.copyOf(model.conditionals[featureIndex], model.conditionals[featureIndex].length);
      for (int labelIndex = 0; labelIndex < model.priors.length; labelIndex++) {
        if (modelType == NaiveBayes.ModelType.Complementary) {
          double nCi = 0;
          double nC = 0;
          for (int j = 0; j < model.priors.length; j++) {
            if (j != labelIndex) {
              nCi += tmp[j];
              nC += labelCounts[j];
            }
          }
          model.conditionals[featureIndex][labelIndex] = modelType.normalize(nCi, model.priors[labelIndex], nC, V);
        } else {
          model.conditionals[featureIndex][labelIndex] = modelType.normalize(model.conditionals[featureIndex][labelIndex], model.priors[labelIndex], labelCounts[labelIndex], V);
        }
      }

    }

    for (int i = 0; i < model.priors.length; i++) {
      model.priors[i] = Math.log(model.priors[i] / N);
    }

    return model;
  }

  @Override
  public void reset() {

  }

}// END OF NaiveBayesLearner
