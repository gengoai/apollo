package com.davidbracewell.apollo.ml.preprocess.filter;

import com.davidbracewell.apollo.ml.Dataset;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.preprocess.InstancePreprocessor;
import com.davidbracewell.function.SerializableDoublePredicate;
import lombok.NonNull;

import java.io.Serializable;
import java.util.stream.Collectors;

/**
 * @author David B. Bracewell
 */
public class ValueFilter implements FilterProcessor<Instance>, InstancePreprocessor, Serializable {
  public static final ValueFilter FINITE = new ValueFilter(Double::isFinite);

  private static final long serialVersionUID = 1L;
  private final SerializableDoublePredicate predicate;

  public ValueFilter(@NonNull SerializableDoublePredicate predicate) {
    this.predicate = predicate;
  }

  @Override
  public void reset() {

  }

  @Override
  public void fit(Dataset<Instance> dataset) {

  }

  @Override
  public boolean trainOnly() {
    return false;
  }

  @Override
  public String describe() {
    return "ValueFilter";
  }

  @Override
  public Instance apply(Instance example) {
    return Instance.create(
      example.getFeatures().stream().filter(f -> predicate.test(f.getValue())).collect(Collectors.toList()),
      example.getLabel()
    );
  }

}// END OF ValueFilter
