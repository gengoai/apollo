package com.davidbracewell.apollo.ml.preprocess.filter;

import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.preprocess.InstancePreprocessor;
import com.davidbracewell.collection.Counter;
import com.davidbracewell.collection.Counters;
import com.davidbracewell.collection.HashMapCounter;
import com.davidbracewell.function.SerializableDoublePredicate;
import lombok.NonNull;

import java.io.Serializable;
import java.util.HashSet;
import java.util.Set;

/**
 * @author David B. Bracewell
 */
public class CountFilter implements FilterProcessor<Instance>, InstancePreprocessor, Serializable {
  private static final long serialVersionUID = 1L;
  private final SerializableDoublePredicate filter;
  private volatile Counter<String> counter = Counters.synchronizedCounter(new HashMapCounter<>());

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
    example.getFeatures().removeIf(f -> !counter.contains(f.getName()));
    return example;
  }

  @Override
  public void reset() {
    counter.clear();
  }

  @Override
  public Set<String> finish(Set<String> removedFeatures) {
    Set<String> removed = new HashSet<>(counter.items());
    counter = counter.filterByKey(f -> !removedFeatures.contains(f)).filterByValue(filter);
    removed.removeAll(counter.items());
    return removed;
  }

  @Override
  public String describe() {
    return "CountFilter: min=" + counter.minimumCount() + ", max=" + counter.maximumCount();
  }

}// END OF CountFilter
