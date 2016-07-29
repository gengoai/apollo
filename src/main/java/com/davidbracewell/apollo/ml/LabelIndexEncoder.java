package com.davidbracewell.apollo.ml;

import com.davidbracewell.apollo.ml.data.Dataset;
import com.davidbracewell.stream.MStream;
import lombok.NonNull;

import java.util.Objects;
import java.util.stream.Collectors;

/**
 * @author David B. Bracewell
 */
public class LabelIndexEncoder extends IndexEncoder implements LabelEncoder {
  private static final long serialVersionUID = 1L;


  @Override
  public void fit(@NonNull Dataset<? extends Example> dataset) {
    if (!isFrozen()) {
      this.index.addAll(
        dataset.stream()
          .parallel()
          .flatMap(ex -> ex.getLabelSpace().map(Object::toString).collect(Collectors.toSet()))
          .filter(Objects::nonNull)
          .distinct()
          .collect()
      );
    }
  }

  @Override
  public void fit(MStream<String> stream) {
    if (!isFrozen()) {
      this.index.addAll(
        stream
          .filter(Objects::nonNull)
          .distinct()
          .collect()
      );
    }
  }

  @Override
  public LabelEncoder createNew() {
    return new LabelIndexEncoder();
  }
}// END OF LabelIndexEncoder
