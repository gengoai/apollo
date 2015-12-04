package com.davidbracewell.apollo.ml.classification.bayes;

import com.davidbracewell.apollo.linalg.DynamicSparseVector;
import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.ml.Dataset;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.classification.ClassifierLearner;
import com.davidbracewell.collection.Collect;
import com.davidbracewell.conversion.Cast;
import com.davidbracewell.function.SerializableIntSupplier;
import lombok.NonNull;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

/**
 * @author David B. Bracewell
 */
public class NaiveBayesLearner extends ClassifierLearner {
  private static final long serialVersionUID = 1L;
  private volatile NaiveBayes.ModelType modelType = NaiveBayes.ModelType.Bernoulli;

  private List<DynamicSparseVector> ensureSize(List<DynamicSparseVector> list, int size, SerializableIntSupplier supplier) {
    while (list.size() <= size) {
      list.add(new DynamicSparseVector(supplier));
    }
    return list;
  }

  @Override
  protected NaiveBayes trainImpl(Dataset<Instance> dataset) {
    switch (modelType) {
      case Bernoulli:
        return bernoulli(dataset);
      case Multinomial:
        return multinomial(dataset);
    }
    throw new IllegalStateException(modelType + " is invalid");
  }

  protected NaiveBayes bernoulli(Dataset<Instance> dataset) {
    NaiveBayes model = new NaiveBayes(
      Cast.as(dataset.labelEncoder()),
      dataset.featureEncoder(),
      dataset.getPreprocessors().getModelProcessors(),
      NaiveBayes.ModelType.Bernoulli
    );

    Iterator<Instance> instanceIterator = dataset.iterator();
    double N = 0;
    DynamicSparseVector priors = new DynamicSparseVector(model::numberOfLabels);
    List<DynamicSparseVector> conditionals = new ArrayList<>();

    while (instanceIterator.hasNext()) {
      Instance instance = instanceIterator.next();
      if (instance.hasLabel()) {
        N++;
        int ci = (int) model.getLabelEncoder().encode(instance.getLabel().toString());
        priors.increment(ci);
        Vector vector = instance.toVector(dataset.featureEncoder());
        for (Vector.Entry entry : Collect.asIterable(vector.nonZeroIterator())) {
          ensureSize(conditionals, entry.index, model::numberOfLabels).get(entry.index).increment(ci);
        }
      }
    }

    model.priors = new double[model.numberOfLabels()];
    model.conditionals = new double[dataset.featureEncoder().size()][model.numberOfLabels()];

    for (int f = 0; f < dataset.featureEncoder().size(); f++) {
      if (conditionals.size() > f) {
        for (int i = 0; i < model.numberOfLabels(); i++) {
          model.conditionals[f][i] = (conditionals.get(f).get(i) + 1) / (priors.get(i) + 2);
        }
      }
    }

    for (int i = 0; i < model.priors.length; i++) {
      model.priors[i] = priors.get(i) / N;
    }

    return model;
  }

  protected NaiveBayes multinomial(Dataset<Instance> dataset) {
    NaiveBayes model = new NaiveBayes(
      Cast.as(dataset.labelEncoder()),
      dataset.featureEncoder(),
      dataset.getPreprocessors().getModelProcessors(),
      NaiveBayes.ModelType.Bernoulli
    );

    Iterator<Instance> instanceIterator = dataset.iterator();
    double N = 0;
    DynamicSparseVector priors = new DynamicSparseVector(model::numberOfLabels);
    DynamicSparseVector labelTotals = new DynamicSparseVector(model::numberOfLabels);
    List<DynamicSparseVector> conditionals = new ArrayList<>();

    while (instanceIterator.hasNext()) {
      Instance instance = instanceIterator.next();
      if (instance.hasLabel()) {
        N++;
        int ci = (int) model.getLabelEncoder().encode(instance.getLabel().toString());
        priors.increment(ci);
        Vector vector = instance.toVector(dataset.featureEncoder());
        for (Vector.Entry entry : Collect.asIterable(vector.nonZeroIterator())) {
          ensureSize(conditionals, entry.index, model::numberOfLabels)
            .get(entry.index)
            .increment(ci, entry.getValue());
          labelTotals.increment(ci, entry.getValue());
        }
      }
    }

    model.priors = new double[model.numberOfLabels()];
    model.conditionals = new double[dataset.featureEncoder().size()][model.numberOfLabels()];

    double V = dataset.featureEncoder().size();
    for (int f = 0; f < dataset.featureEncoder().size(); f++) {
      if (conditionals.size() > f) {
        for (int i = 0; i < model.numberOfLabels(); i++) {
          model.conditionals[f][i] = (conditionals.get(f).get(i) + 1) / (labelTotals.get(i) + V);
        }
      }
    }

    for (int i = 0; i < model.priors.length; i++) {
      model.priors[i] = priors.get(i) / N;
    }

    return model;
  }


  public NaiveBayes.ModelType getModelType() {
    return modelType;
  }

  public void setModelType(@NonNull NaiveBayes.ModelType modelType) {
    this.modelType = modelType;
  }
}// END OF NaiveBayesLearner
