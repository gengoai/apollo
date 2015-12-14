package com.davidbracewell.apollo.ml.preprocess.filter;

import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.preprocess.FilterProcessor;
import com.davidbracewell.apollo.ml.preprocess.InstancePreprocessor;
import com.davidbracewell.function.SerializableDoublePredicate;

import java.io.Serializable;
import java.util.stream.Collectors;

/**
 * @author David B. Bracewell
 */
public class ValueFilter implements FilterProcessor<Instance>, InstancePreprocessor, Serializable {
  private static final long serialVersionUID = 1L;
  private final SerializableDoublePredicate predicate;

  public ValueFilter(SerializableDoublePredicate predicate) {
    this.predicate = predicate;
  }

  @Override
  public void visit(Instance example) {

  }

  @Override
  public Instance process(Instance example) {
    return Instance.create(
      example.getFeatures().stream().filter(f -> predicate.test(f.getValue())).collect(Collectors.toList()),
      example.getLabel()
    );
  }

  @Override
  public void finish() {

  }

  @Override
  public void reset() {

  }
}// END OF ValueFilter
