package com.davidbracewell.apollo.ml.preprocess.filter;

import com.davidbracewell.apollo.ml.Feature;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.preprocess.RestrictedInstancePreprocessor;
import com.davidbracewell.collection.HashMapCounter;
import com.davidbracewell.stream.MStream;
import lombok.NonNull;

import java.io.Serializable;
import java.util.Collections;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * @author David B. Bracewell
 */
public class MinCountFilter extends RestrictedInstancePreprocessor implements FilterProcessor<Instance>, Serializable {
  private static final long serialVersionUID = 1L;
  private final long minCount;
  private volatile Set<String> selectedFeatures = Collections.emptySet();

  public MinCountFilter(@NonNull String featurePrefix, long minCount) {
    super(featurePrefix);
    this.minCount = minCount;
  }

  public MinCountFilter(long minCount) {
    super(null);
    this.minCount = minCount;
  }

  @Override
  protected void restrictedFitImpl(MStream<List<Feature>> stream) {
    selectedFeatures = new HashMapCounter<>(
      stream.flatMap(l -> l.stream().map(Feature::getName).collect(Collectors.toList()))
        .countByValue()
    )
      .filterByValue(v -> v >= minCount)
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
      return "CountFilter: minCount=" + minCount;
    }
    return "CountFilter[" + getRestriction() + "]: minCount=" + minCount;
  }

}// END OF CountFilter
