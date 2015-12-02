package com.davidbracewell.apollo.ml;

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
public abstract class BaseDataset implements Dataset, Serializable {

  private final FeatureEncoder featureEncoder;
  private final LabelEncoder labelEncoder;

  protected BaseDataset(FeatureEncoder featureEncoder, LabelEncoder labelEncoder) {
    this.featureEncoder = featureEncoder;
    this.labelEncoder = labelEncoder;
  }

  @Override
  public void addAll(@NonNull MStream<Instance> stream) {
    stream.forEach(this::add);
  }

  protected final Dataset create(@NonNull MStream<Instance> instances) {
    return create(instances, getFeatureEncoder().createNew(), getLabelEncoder().createNew());
  }

  protected abstract Dataset create(@NonNull MStream<Instance> instances, @NonNull FeatureEncoder featureEncoder, @NonNull LabelEncoder labelEncoder);

  protected MStream<Instance> slice(int start, int end) {
    return stream().skip(Math.max(0, start - 1)).limit(end - start);
  }

  @Override
  public void addAll(Iterable<Instance> instances) {
    if (instances != null) {
      instances.forEach(this::add);
    }
  }

  @Override
  public Tuple2<Dataset, Dataset> split(double pctTrain) {
    int split = (int) Math.floor(pctTrain * size());
    return Tuple2.of(
      create(slice(0, split)),
      create(slice(split, size()))
    );
  }

  @Override
  public Dataset copy() {
    return create(stream().map(Instance::copy));
  }

  @Override
  public List<Tuple2<Dataset, Dataset>> fold(int numberOfFolds) {
    Preconditions.checkArgument(numberOfFolds > 0, "Number of folds must be >= 0");
    ArrayList<Tuple2<Dataset, Dataset>> folds = new ArrayList<>();

    int foldSize = size() / numberOfFolds;
    for (int i = 0; i < numberOfFolds; i++) {
      MStream<Instance> train;
      MStream<Instance> test;
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
  public Dataset sample(int sampleSize) {
    return create(stream().sample(sampleSize));
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
