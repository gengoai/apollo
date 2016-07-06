package com.davidbracewell.apollo.ml.preprocess.filter;

import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.preprocess.InstancePreprocessor;
import com.davidbracewell.function.SerializableDoublePredicate;

import java.io.Serializable;
import java.util.Collections;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * @author David B. Bracewell
 */
public class ValueFilter implements FilterProcessor<Instance>, InstancePreprocessor, Serializable {
  public static final ValueFilter FINITE = new ValueFilter(Double::isFinite);

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
  public Set<String> finish(Set<String> removedFeatures) {
    return Collections.emptySet();
  }

  @Override
  public void reset() {

  }

  @Override
  public boolean trainOnly() {
    return false;
  }

  @Override
  public String describe() {
    return "ValueFilter";
  }

}// END OF ValueFilter
