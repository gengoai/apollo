package com.davidbracewell.apollo.ml.preprocess.filter;

import com.davidbracewell.apollo.ml.Feature;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.preprocess.RestrictedInstancePreprocessor;
import com.davidbracewell.collection.HashMapCounter;
import com.davidbracewell.stream.MStream;
import lombok.NonNull;
import org.apache.commons.lang.math.LongRange;

import java.io.Serializable;
import java.util.Collections;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * @author David B. Bracewell
 */
public class CountFilter extends RestrictedInstancePreprocessor implements FilterProcessor<Instance>, Serializable {
  private static final long serialVersionUID = 1L;
  private final LongRange range;
  private volatile Set<String> selectedFeatures = Collections.emptySet();

  public CountFilter(@NonNull String featurePrefix, @NonNull LongRange filter) {
    super(featurePrefix);
    this.range = filter;
  }

  public CountFilter(@NonNull LongRange filter) {
    super(null);
    this.range = filter;
  }

  @Override
  protected void restrictedFitImpl(MStream<List<Feature>> stream) {
    selectedFeatures = new HashMapCounter<>(
      stream.flatMap(l -> l.stream().map(Feature::getName).collect(Collectors.toList()))
        .countByValue()
    )
      .filterByValue(range::containsDouble)
      .items();
  }

  @Override
  public void reset() {
    selectedFeatures.clear();
  }

  @Override
  protected Stream<Feature> restrictedProcessImpl(Stream<Feature> featureStream, Instance originalExample) {
    return featureStream.filter(f -> selectedFeatures.contains(f.getName()));
  }

  @Override
  public String describe() {
    if (acceptAll()) {
      return "CountFilter: min=" + range.getMinimumLong() + ", max=" + range.getMaximumLong();
    }
    return "CountFilter[" + getRestriction() + "]: min=" + range.getMinimumLong() + ", max=" + range.getMaximumLong();
  }

}// END OF CountFilter
