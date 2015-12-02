package com.davidbracewell.apollo.ml;

import com.davidbracewell.apollo.ml.classification.Classifier;
import com.davidbracewell.apollo.ml.classification.bayes.BernoulliNaiveBayesLearner;
import com.davidbracewell.stream.MStream;
import com.davidbracewell.stream.Streams;
import com.davidbracewell.string.StringUtils;
import com.google.common.util.concurrent.AtomicDouble;
import lombok.EqualsAndHashCode;
import lombok.NonNull;
import lombok.ToString;

import java.io.Serializable;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * @author David B. Bracewell
 */
@EqualsAndHashCode(callSuper = false)
@ToString
public class InMemoryDataset extends BaseDataset implements Serializable {

  private final List<Instance> instances = new LinkedList<>();

  public InMemoryDataset(@NonNull FeatureEncoder featureEncoder, @NonNull LabelEncoder labelEncoder) {
    super(featureEncoder, labelEncoder);
  }



  public static Set<String> split(String s) {
    return s.chars().mapToObj(i -> Character.toString((char) i)).collect(Collectors.toSet());
  }

  @Override
  public void add(Instance instance) {
    if (instance != null) {
      instance.forEach(f -> getFeatureEncoder().encode(f.getName()));
      if (instance.hasLabel()) {
        getLabelEncoder().encode(instance.getLabel());
      }
      instances.add(instance);
    }
  }

  @Override
  public int size() {
    return instances.size();
  }

  @Override
  public MStream<Instance> stream() {
    return Streams.of(instances, false);
  }

  @Override
  public Dataset shuffle() {
    Collections.shuffle(instances);
    return this;
  }

  @Override
  protected Dataset create(@NonNull MStream<Instance> instances, @NonNull FeatureEncoder featureEncoder, @NonNull LabelEncoder labelEncoder) {
    Dataset dataset = new InMemoryDataset(featureEncoder, labelEncoder);
    dataset.addAll(instances.collect());
    return dataset;
  }
}// END OF InMemoryDataset
