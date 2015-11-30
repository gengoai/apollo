package com.davidbracewell.apollo.learning.classification.bayes;

import com.davidbracewell.apollo.learning.FeatureEncoder;
import com.davidbracewell.apollo.learning.Featurizer;
import com.davidbracewell.apollo.learning.Instance;
import com.davidbracewell.apollo.learning.classification.ClassifierLearner;
import com.davidbracewell.apollo.learning.classification.OnlineClassifierTrainer;
import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.collection.Collect;
import com.davidbracewell.collection.Index;
import com.davidbracewell.collection.Indexes;
import com.davidbracewell.function.SerializableSupplier;
import com.davidbracewell.stream.MStream;
import com.davidbracewell.stream.Streams;
import lombok.NonNull;

import java.util.Iterator;
import java.util.List;
import java.util.function.Supplier;

/**
 * @author David B. Bracewell
 */
public class BernoulliNaiveBayesLearner<T> implements ClassifierLearner<T>, OnlineClassifierTrainer<T> {

  private final SerializableSupplier<FeatureEncoder> featureEncoderSupplier;
  private final Featurizer<T> featurizer;

  public BernoulliNaiveBayesLearner(@NonNull SerializableSupplier<FeatureEncoder> featureEncoderSupplier, Featurizer<T> featurizer) {
    this.featureEncoderSupplier = featureEncoderSupplier;
    this.featurizer = featurizer;
  }

  @Override
  public NaiveBayes<T> train(@NonNull List<Instance> instanceList) {
    return train(() -> Streams.of(instanceList, false));
  }

  @Override
  public NaiveBayes<T> train(Supplier<MStream<Instance>> instanceSupplier) {
    Index<String> classLabels = Indexes.newIndex();
    FeatureEncoder featureEncoder = featureEncoderSupplier.get();
    NaiveBayes<T> model = new BernoulliNaiveBayes<T>(classLabels, featureEncoder, featurizer);
    Iterator<Instance> instanceIterator = instanceSupplier.get().iterator();
    double N = 0;
    instanceSupplier.get().forEach(instance -> {
      classLabels.add(instance.getLabel().toString());
      featureEncoder.toVector(instance);
    });
    model.priors = new double[classLabels.size()];
    model.conditionals = new double[featureEncoder.size()][classLabels.size()];
    while (instanceIterator.hasNext()) {
      Instance instance = instanceIterator.next();
      if (instance.hasLabel()) {
        N++;
        int ci = classLabels.add(instance.getLabel().toString());
        model.priors[ci]++;
        Vector vector = featureEncoder.toVector(instance);
        for (Vector.Entry entry : Collect.asIterable(vector.nonZeroIterator())) {
          model.conditionals[entry.index][ci]++;
        }
      }
    }

    for (int f = 0; f < featureEncoder.size(); f++) {
      for (int i = 0; i < classLabels.size(); i++) {
        model.conditionals[f][i] = (model.conditionals[f][i] + 1) / (model.priors[i] + 2);
      }
    }

    for (int i = 0; i < model.priors.length; i++) {
      model.priors[i] = model.priors[i] / N;
    }

    return model;
  }

}// END OF BernoulliNaiveBayesLearner
