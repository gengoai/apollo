package com.davidbracewell.apollo.ml.preprocess;

import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.collection.Counter;
import com.davidbracewell.collection.Counters;
import com.davidbracewell.function.SerializableDoublePredicate;
import lombok.NonNull;

import java.io.Serializable;
import java.util.stream.Collectors;

/**
 * @author David B. Bracewell
 */
public class CountFilter implements FilterProcessor<Instance>, Serializable {
  private static final long serialVersionUID = 1L;
  private final SerializableDoublePredicate filter;
  private volatile Counter<String> counter = Counters.newConcurrentCounter();

  public CountFilter(@NonNull SerializableDoublePredicate filter) {
    this.filter = filter;
  }

  @Override
  public void visit(Instance example) {
    if (example != null) {
      example.forEach(f -> counter.increment(f.getName()));
    }
  }

  @Override
  public Instance process(@NonNull Instance example) {
    return Instance.create(
      example.stream().filter(f -> counter.contains(f.getName())).collect(Collectors.toList()),
      example.getLabel()
    );
  }

  @Override
  public void reset() {
    counter.clear();
  }

  @Override
  public void finish() {
    counter = counter.filterByValue(filter);
  }

}// END OF CountFilter
