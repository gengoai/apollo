package com.davidbracewell.apollo.ml;

import com.davidbracewell.apollo.ml.data.Dataset;
import com.davidbracewell.stream.accumulator.CollectionAccumulatable;
import com.davidbracewell.stream.accumulator.MAccumulator;
import lombok.NonNull;

import java.util.HashSet;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * @author David B. Bracewell
 */
public class LabelIndexEncoder extends IndexEncoder implements LabelEncoder {
  private static final long serialVersionUID = 1L;


  @Override
  public void fit(@NonNull Dataset<? extends Example> dataset) {
    if (!isFrozen()) {
      MAccumulator<Set<String>> accumulator = dataset.getStreamingContext().accumulator(new HashSet<>(), new CollectionAccumulatable<>());
      dataset.stream().forEach(ex -> accumulator.add(ex.getLabelSpace().map(Object::toString).collect(Collectors.toSet())));
      this.index.addAll(accumulator.value());
    }
  }

  @Override
  public LabelEncoder createNew() {
    return new LabelIndexEncoder();
  }
}// END OF LabelIndexEncoder
