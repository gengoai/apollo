package com.davidbracewell.apollo.learning.classification.bayes;

import com.davidbracewell.apollo.learning.FeatureEncoder;
import com.davidbracewell.apollo.learning.Featurizer;
import com.davidbracewell.apollo.learning.Instance;
import com.davidbracewell.apollo.learning.classification.ClassifierLearner;
import com.davidbracewell.apollo.learning.classification.OnlineClassifierTrainer;
import com.davidbracewell.collection.Index;
import com.davidbracewell.collection.Indexes;
import com.davidbracewell.function.SerializableSupplier;
import com.davidbracewell.stream.MStream;
import com.davidbracewell.stream.Streams;
import lombok.NonNull;
import org.apache.commons.math3.linear.RealVector;

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
    while (instanceIterator.hasNext()) {
      Instance instance = instanceIterator.next();
      if (instance.hasLabel()) {
        N++;
        int ci = classLabels.add(instance.getLabel().toString());
        model.priors[ci]++;
        RealVector vector = featureEncoder.toVector(instance);
        Iterator<RealVector.Entry> iterator = vector.sparseIterator();

      }
    }
    return model;
  }

}// END OF BernoulliNaiveBayesLearner
