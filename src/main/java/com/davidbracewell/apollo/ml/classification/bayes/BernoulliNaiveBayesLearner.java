package com.davidbracewell.apollo.ml.classification.bayes;

import com.davidbracewell.apollo.ml.FeatureEncoder;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.classification.ClassifierLearner;
import com.davidbracewell.apollo.ml.classification.OnlineClassifierTrainer;
import com.davidbracewell.apollo.linalg.DynamicSparseVector;
import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.collection.Collect;
import com.davidbracewell.collection.Index;
import com.davidbracewell.collection.Indexes;
import com.davidbracewell.function.SerializableIntSupplier;
import com.davidbracewell.function.SerializableSupplier;
import com.davidbracewell.stream.MStream;
import com.davidbracewell.stream.Streams;
import lombok.NonNull;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.function.Supplier;

/**
 * @author David B. Bracewell
 */
public class BernoulliNaiveBayesLearner<T> implements ClassifierLearner<T>, OnlineClassifierTrainer<T> {

  private final SerializableSupplier<FeatureEncoder> featureEncoderSupplier;

  public BernoulliNaiveBayesLearner(@NonNull SerializableSupplier<FeatureEncoder> featureEncoderSupplier) {
    this.featureEncoderSupplier = featureEncoderSupplier;
  }

  @Override
  public NaiveBayes train(@NonNull List<Instance> instanceList) {
    return train(() -> Streams.of(instanceList, false));
  }


  private List<DynamicSparseVector> ensureSize(List<DynamicSparseVector> list, int size, SerializableIntSupplier supplier) {
    while (list.size() <= size) {
      list.add(new DynamicSparseVector(supplier));
    }
    return list;
  }

  @Override
  public NaiveBayes train(Supplier<MStream<Instance>> instanceSupplier) {
    Index<String> classLabels = Indexes.newIndex();
    FeatureEncoder featureEncoder = featureEncoderSupplier.get();
    NaiveBayes model = new BernoulliNaiveBayes(classLabels, featureEncoder);

    Iterator<Instance> instanceIterator = instanceSupplier.get().iterator();
    double N = 0;
    DynamicSparseVector priors = new DynamicSparseVector(classLabels::size);
    List<DynamicSparseVector> conditionals = new ArrayList<>();

    while (instanceIterator.hasNext()) {
      Instance instance = instanceIterator.next();
      if (instance.hasLabel()) {
        N++;
        int ci = classLabels.add(instance.getLabel().toString());
        priors.increment(ci);
        Vector vector = featureEncoder.toVector(instance);
        for (Vector.Entry entry : Collect.asIterable(vector.nonZeroIterator())) {
          ensureSize(conditionals, entry.index, classLabels::size).get(entry.index).increment(ci);
        }
      }
    }

    model.priors = new double[classLabels.size()];
    model.conditionals = new double[featureEncoder.size()][classLabels.size()];

    for (int f = 0; f < featureEncoder.size(); f++) {
      for (int i = 0; i < classLabels.size(); i++) {
        model.conditionals[f][i] = (conditionals.get(f).get(i) + 1) / (priors.get(i) + 2);
      }
    }

    for (int i = 0; i < model.priors.length; i++) {
      model.priors[i] = priors.get(i) / N;
    }

    model.getFeatureEncoder().freeze();
    return model;
  }

}// END OF BernoulliNaiveBayesLearner
