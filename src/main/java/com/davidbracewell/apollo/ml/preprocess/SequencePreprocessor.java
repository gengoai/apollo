package com.davidbracewell.apollo.ml.preprocess;

import com.davidbracewell.apollo.ml.Dataset;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.MStreamDataset;
import com.davidbracewell.apollo.ml.sequence.Sequence;
import lombok.NonNull;

import java.io.Serializable;
import java.util.stream.Collectors;

/**
 * @author David B. Bracewell
 */
public class SequencePreprocessor implements Preprocessor<Sequence>, Serializable {
  private static final long serialVersionUID = 1L;
  private final Preprocessor<Instance> instancePreprocessor;

  public SequencePreprocessor(@NonNull Preprocessor<Instance> instancePreprocessor) {
    this.instancePreprocessor = instancePreprocessor;
  }

  @Override
  public void fit(Dataset<Sequence> dataset) {
    instancePreprocessor.fit(
      new MStreamDataset<>(
        dataset.getFeatureEncoder(),
        dataset.getLabelEncoder(),
        PreprocessorList.empty(),
        dataset.stream().flatMap(Sequence::asInstances)
      )
    );
  }

  @Override
  public Sequence apply(Sequence example) {
    return new Sequence(example.asInstances().stream().map(instancePreprocessor::apply).collect(Collectors.toList()));
  }

  @Override
  public boolean trainOnly() {
    return instancePreprocessor.trainOnly();
  }

  @Override
  public void reset() {
    instancePreprocessor.reset();
  }

  @Override
  public String describe() {
    return instancePreprocessor.describe();
  }

}// END OF SequencePreprocessor
