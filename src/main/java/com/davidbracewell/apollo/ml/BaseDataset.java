package com.davidbracewell.apollo.ml;

import com.davidbracewell.conversion.Cast;
import com.davidbracewell.stream.MStream;
import com.davidbracewell.tuple.Tuple2;
import com.google.common.base.Preconditions;
import lombok.NonNull;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

/**
 * @author David B. Bracewell
 */
public abstract class BaseDataset<T extends Example> implements Dataset<T>, Serializable {

  private final Encoder featureEncoder;
  private final Encoder labelEncoder;

  protected BaseDataset(Encoder featureEncoder, Encoder labelEncoder) {
    this.featureEncoder = featureEncoder;
    this.labelEncoder = labelEncoder;
  }

  @Override
  public void addAll(@NonNull MStream<T> stream) {
    stream.forEach(this::add);
  }

  protected final Dataset<T> create(@NonNull MStream<T> instances) {
    return create(instances, featureEncoder().createNew(), labelEncoder().createNew());
  }

  protected abstract Dataset<T> create(@NonNull MStream<T> instances, @NonNull Encoder featureEncoder, @NonNull Encoder labelEncoder);

  protected MStream<T> slice(int start, int end) {
    return stream().skip(Math.max(0, start - 1)).limit(end - start);
  }

  @Override
  public void addAll(Iterable<T> instances) {
    if (instances != null) {
      instances.forEach(this::add);
    }
  }

  @Override
  public Tuple2<Dataset<T>, Dataset<T>> split(double pctTrain) {
    int split = (int) Math.floor(pctTrain * size());
    return Tuple2.of(
      create(slice(0, split)),
      create(slice(split, size()))
    );
  }

  @Override
  public Dataset<T> copy() {
    return create(stream().map(e -> Cast.as(e.copy())));
  }

  @Override
  public List<Tuple2<Dataset<T>, Dataset<T>>> fold(int numberOfFolds) {
    Preconditions.checkArgument(numberOfFolds > 0, "Number of folds must be >= 0");
    ArrayList<Tuple2<Dataset<T>, Dataset<T>>> folds = new ArrayList<>();

    int foldSize = size() / numberOfFolds;
    for (int i = 0; i < numberOfFolds; i++) {
      MStream<T> train;
      MStream<T> test;
      if (i == 0) {
        test = slice(0, foldSize);
        train = slice(foldSize, size());
      } else if (i == numberOfFolds - 1) {
        test = slice(size() - foldSize, size());
        train = slice(0, size() - foldSize);
      } else {
        train = slice(0, foldSize * i).union(slice(foldSize * i + foldSize, size()));
        test = slice(foldSize * i, foldSize * i + foldSize);
      }

      folds.add(
        Tuple2.of(create(train), create(test))
      );
    }

    folds.trimToSize();
    return folds;
  }

  @Override
  public Dataset<T> sample(int sampleSize) {
    return create(stream().sample(sampleSize));
  }

  @Override
  public Encoder featureEncoder() {
    return featureEncoder;
  }

  @Override
  public Encoder labelEncoder() {
    return labelEncoder;
  }

}// END OF BaseDataset
