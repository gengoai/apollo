package com.davidbracewell.apollo.ml;

import com.davidbracewell.stream.MStream;

import java.io.Serializable;

/**
 * @author David B. Bracewell
 */
public abstract class BaseDataset implements Dataset, Serializable {

  private FeatureEncoder featureEncoder;
  private LabelEncoder labelEncoder;

  @Override
  public void addAll(Iterable<Instance> instances) {
    if (instances != null) {
      instances.forEach(this::add);
    }
  }

  @Override
  public MStream<Instance> stream() {
    return null;
  }

  @Override
  public FeatureEncoder getFeatureEncoder() {
    return featureEncoder;
  }

  @Override
  public LabelEncoder getLabelEncoder() {
    return labelEncoder;
  }

}// END OF BaseDataset
