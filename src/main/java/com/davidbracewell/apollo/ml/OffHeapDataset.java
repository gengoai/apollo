package com.davidbracewell.apollo.ml;

import com.davidbracewell.apollo.ml.preprocess.PreprocessorList;
import com.davidbracewell.collection.Collect;
import com.davidbracewell.conversion.Cast;
import com.davidbracewell.function.Unchecked;
import com.davidbracewell.io.Resources;
import com.davidbracewell.io.resource.Resource;
import com.davidbracewell.stream.MStream;
import com.davidbracewell.stream.StreamingContext;
import com.google.common.base.Throwables;
import lombok.NonNull;

import java.io.BufferedWriter;
import java.io.IOException;
import java.util.Random;
import java.util.concurrent.atomic.AtomicLong;

/**
 * @author David B. Bracewell
 */
public class OffHeapDataset<T extends Example> extends Dataset<T> {
  private static final long serialVersionUID = 1L;
  private final AtomicLong id = new AtomicLong();
  private Resource outputResource = Resources.temporaryDirectory();
  private Class<T> clazz;
  private int size = 0;

  protected OffHeapDataset(Encoder featureEncoder, Encoder labelEncoder, PreprocessorList<T> preprocessors) {
    super(featureEncoder, labelEncoder, preprocessors);
  }

  @Override
  protected void addAll(@NonNull MStream<T> instances) {
    try (BufferedWriter writer = new BufferedWriter(outputResource.getChild("part-" + id.incrementAndGet() + ".json").writer())) {
      for (T instance : Collect.asIterable(instances.iterator())) {
        clazz = Cast.as(instance.getClass());
        writer.write(instance.toJson());
        writer.newLine();
        size++;
      }
    } catch (IOException e) {
      throw Throwables.propagate(e);
    }
  }

  @Override
  public DatasetType getType() {
    return DatasetType.OffHeap;
  }

  @Override
  public MStream<T> stream() {
    return StreamingContext.local().stream(
      outputResource.getChildren().stream()
        .flatMap(Unchecked.function(r -> r.readLines().stream()))
        .map(Unchecked.function(line -> Cast.as(Example.fromJson(line, clazz))))
    );
  }

  @Override
  public Dataset<T> shuffle(@NonNull Random random) {
    return create(stream().shuffle(), getFeatureEncoder().createNew(), getLabelEncoder().createNew(), null);
  }

  @Override
  public int size() {
    return size;
  }

  @Override
  protected Dataset<T> create(@NonNull MStream<T> instances, @NonNull Encoder featureEncoder, @NonNull Encoder labelEncoder, PreprocessorList<T> preprocessors) {
    Dataset<T> dataset = new OffHeapDataset<>(featureEncoder, labelEncoder, preprocessors);
    dataset.addAll(instances);
    return dataset;
  }

}// END OF OffHeapDataset
