package com.davidbracewell.apollo.ml.preprocess;

import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.function.SerializablePredicate;
import lombok.NonNull;

import java.io.Serializable;
import java.util.stream.Collectors;

/**
 * @author David B. Bracewell
 */
public class NameFilter implements FilterProcessor<Instance>, Serializable {
  private static final long serialVersionUID = 1L;
  private final SerializablePredicate<String> filter;

  public NameFilter(@NonNull SerializablePredicate<String> filter) {
    this.filter = filter;
  }

  @Override
  public void visit(Instance example) {

  }

  @Override
  public Instance process(Instance example) {
    return Instance.create(
      example.stream().filter(f -> filter.test(f.getName())).collect(Collectors.toList()),
      example.getLabel()
    );
  }

  @Override
  public void finish() {

  }

  @Override
  public void reset() {

  }

}// END OF NameFilter
