package com.davidbracewell.apollo.ml.preprocess;

import com.davidbracewell.apollo.ml.Encoder;
import com.davidbracewell.apollo.ml.Instance;
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

  public SequencePreprocessor(Preprocessor<Instance> instancePreprocessor) {
    this.instancePreprocessor = instancePreprocessor;
  }

  @Override
  public void visit(Sequence example) {
    if (example != null) {
      example.asInstances().forEach(instancePreprocessor::visit);
    }
  }

  @Override
  public Sequence process(@NonNull Sequence example) {
    return new Sequence(example.asInstances().stream().map(instancePreprocessor::process).collect(Collectors.toList()));
  }

  @Override
  public void finish() {
    instancePreprocessor.finish();
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
  public void trimToSize(Encoder encoder) {
    instancePreprocessor.trimToSize(encoder);
  }

}// END OF SequencePreprocessor
