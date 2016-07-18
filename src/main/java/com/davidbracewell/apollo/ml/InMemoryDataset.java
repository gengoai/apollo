package com.davidbracewell.apollo.ml;

import com.davidbracewell.apollo.ml.preprocess.Preprocessor;
import com.davidbracewell.apollo.ml.preprocess.PreprocessorList;
import com.davidbracewell.collection.Collect;
import com.davidbracewell.collection.Interner;
import com.davidbracewell.conversion.Cast;
import com.davidbracewell.stream.MStream;
import com.davidbracewell.stream.StreamingContext;
import lombok.EqualsAndHashCode;
import lombok.NonNull;
import lombok.ToString;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Random;

/**
 * <p>A Dataset that stores all examples in memory. Feature names are interned to conserve memory. In addition, methods
 * for adding examples are public.</p>
 *
 * @param <T> the type parameter
 * @author David B. Bracewell
 */
@EqualsAndHashCode(callSuper = false)
@ToString
public class InMemoryDataset<T extends Example> extends Dataset<T> {

  private static final Interner<String> interner = new Interner<>();
  private final List<T> instances = new ArrayList<>();

  /**
   * Instantiates a new In memory dataset.
   *
   * @param featureEncoder the feature encoder
   * @param labelEncoder   the label encoder
   * @param preprocessors  the preprocessors
   */
  public InMemoryDataset(Encoder featureEncoder, Encoder labelEncoder, PreprocessorList<T> preprocessors) {
    super(featureEncoder, labelEncoder, preprocessors);
  }

  @Override
  public DatasetType getType() {
    return DatasetType.InMemory;
  }

  /**
   * Adds an example
   *
   * @param example the example
   */
  public void add(T example) {
    if (example != null) {
      getLabelEncoder().encode(example.getLabelSpace());
      instances.add(example);
    }
  }

  @Override
  public void close() {
    instances.clear();
  }

  @Override
  protected Iterator<T> rawIterator() {
    return instances.iterator();
  }

  @Override
  protected void addAll(@NonNull Collection<T> instances) {
    super.addAll(instances);
  }

  @Override
  protected void addAll(@NonNull MStream<T> stream) {
    for (T instance : Collect.asIterable(stream.iterator())) {
      getLabelEncoder().encode(instance.getLabelSpace());
      instances.add(Cast.as(instance.intern(interner)));
    }
  }

  @Override
  protected MStream<T> slice(int start, int end) {
    return StreamingContext.local().stream(instances.subList(start, end).stream());
  }

  @Override
  public int size() {
    return instances.size();
  }

  @Override
  public MStream<T> stream() {
    return StreamingContext.local().stream(instances).map(getEncoderPair()::encode);
  }

  @Override
  public Dataset<T> shuffle(@NonNull Random random) {
    Collections.shuffle(instances, random);
    return this;
  }

  @Override
  protected Dataset<T> create(@NonNull MStream<T> instances, @NonNull Encoder featureEncoder, @NonNull Encoder labelEncoder, PreprocessorList<T> preprocessors) {
    Dataset<T> dataset = new InMemoryDataset<>(featureEncoder, labelEncoder, preprocessors);
    dataset.addAll(instances);
    return dataset;
  }

}// END OF InMemoryDataset
