package com.davidbracewell.apollo.ml;

import com.davidbracewell.apollo.ml.preprocess.PreprocessorList;
import com.davidbracewell.collection.Collect;
import com.davidbracewell.collection.Interner;
import com.davidbracewell.conversion.Cast;
import com.davidbracewell.stream.MStream;
import com.davidbracewell.stream.Streams;
import lombok.EqualsAndHashCode;
import lombok.NonNull;
import lombok.ToString;

import java.util.Collections;
import java.util.LinkedList;
import java.util.List;

/**
 * @author David B. Bracewell
 */
@EqualsAndHashCode(callSuper = false)
@ToString
public class InMemoryDataset<T extends Example> extends Dataset<T> {

  private static final Interner<String> interner = new Interner<>();
  private final List<T> instances = new LinkedList<>();

  public InMemoryDataset(Encoder featureEncoder, Encoder labelEncoder, PreprocessorList<T> preprocessors) {
    super(featureEncoder, labelEncoder, preprocessors);
  }

  @Override
  protected void addAll(@NonNull MStream<T> stream) {
    for (T instance : Collect.asIterable(stream.iterator())) {
      instances.add(Cast.as(instance.intern(interner)));
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
  protected Dataset<T> create(@NonNull MStream<T> instances, @NonNull Encoder featureEncoder, @NonNull Encoder labelEncoder, PreprocessorList<T> preprocessors) {
    Dataset<T> dataset = new InMemoryDataset<>(featureEncoder, labelEncoder, preprocessors);
    dataset.addAll(instances);
    return dataset;
  }


}// END OF InMemoryDataset
