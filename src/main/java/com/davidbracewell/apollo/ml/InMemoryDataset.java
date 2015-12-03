package com.davidbracewell.apollo.ml;

import com.davidbracewell.stream.MStream;
import com.davidbracewell.stream.Streams;
import lombok.EqualsAndHashCode;
import lombok.NonNull;
import lombok.ToString;

import java.io.Serializable;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;

/**
 * @author David B. Bracewell
 */
@EqualsAndHashCode(callSuper = false)
@ToString
public class InMemoryDataset<T extends Example> extends BaseDataset<T> implements Serializable {

  private final List<T> instances = new LinkedList<>();

  public InMemoryDataset(@NonNull Encoder featureEncoder, @NonNull Encoder labelEncoder) {
    super(featureEncoder, labelEncoder);
  }

  @Override
  public void add(T instance) {
    if (instance != null) {
      featureEncoder().encode(instance.getFeatureSpace());
      labelEncoder().encode(instance.getLabelSpace());
      instances.add(instance);
    }
  }

  @Override
  public int size() {
    return instances.size();
  }

  @Override
  public MStream<T> stream() {
    return Streams.of(instances, false);
  }

  @Override
  public Dataset<T> shuffle() {
    Collections.shuffle(instances);
    return this;
  }

  @Override
  protected Dataset<T> create(@NonNull MStream<T> instances, @NonNull Encoder featureEncoder, @NonNull Encoder labelEncoder) {
    Dataset<T> dataset = new InMemoryDataset<>(featureEncoder, labelEncoder);
    dataset.addAll(instances.collect());
    return dataset;
  }
}// END OF InMemoryDataset
