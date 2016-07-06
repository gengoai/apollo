package com.davidbracewell.apollo.ml.preprocess.transform;

import com.davidbracewell.apollo.ml.Encoder;
import com.davidbracewell.apollo.ml.Feature;
import com.davidbracewell.collection.EnhancedDoubleStatistics;
import com.davidbracewell.string.StringUtils;

import java.util.Collections;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.stream.Stream;

/**
 * @author David B. Bracewell
 */
public class ZScoreTransform extends RestrictedTransform {
  private static final long serialVersionUID = 1L;
  private final EnhancedDoubleStatistics statistics = new EnhancedDoubleStatistics();
  private volatile AtomicBoolean finished = new AtomicBoolean(false);

  public ZScoreTransform() {
    super(StringUtils.EMPTY);
  }

  public ZScoreTransform(String featureNamePrefix) {
    super(featureNamePrefix);
  }

  @Override
  protected void visitImpl(Stream<Feature> featureStream) {
    if (!finished.get()) {
      featureStream.mapToDouble(Feature::getValue).forEach(statistics::accept);
    }
  }

  @Override
  protected Stream<Feature> processImpl(Stream<Feature> featureStream) {
    return featureStream.map(feature -> Feature.real(feature.getName(), (feature.getValue() - statistics.getAverage()) / statistics.getSampleStandardDeviation()));
  }

  @Override
  public Set<String> finish(Set<String> removedFeatures) {
    finished.set(true);
    return Collections.emptySet();
  }

  @Override
  public void reset() {
    finished.set(false);
    statistics.clear();
  }

  @Override
  public void trimToSize(Encoder encoder) {

  }

}// END OF ZScoreTransform
