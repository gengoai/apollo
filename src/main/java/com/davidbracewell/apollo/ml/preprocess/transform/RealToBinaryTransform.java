package com.davidbracewell.apollo.ml.preprocess.transform;

import com.davidbracewell.apollo.ml.Encoder;
import com.davidbracewell.apollo.ml.Feature;
import com.davidbracewell.string.StringUtils;

import java.util.Collections;
import java.util.Set;
import java.util.stream.Stream;

/**
 * @author David B. Bracewell
 */
public class RealToBinaryTransform extends RestrictedTransform {
  private static final long serialVersionUID = 1L;
  private final double threshold;

  public RealToBinaryTransform(double threshold) {
    this(StringUtils.EMPTY, threshold);
  }

  public RealToBinaryTransform(String featureNamePrefix, double threshold) {
    super(featureNamePrefix);
    this.threshold = threshold;
  }

  @Override
  protected void visitImpl(Stream<Feature> featureStream) {
  }

  @Override
  protected Stream<Feature> processImpl(Stream<Feature> featureStream) {
    return featureStream.filter(f -> f.getValue() >= threshold).map(feature -> Feature.TRUE(feature.getName()));
  }

  @Override
  public Set<String> finish(Set<String> removedFeatures) {
    return Collections.emptySet();
  }

  @Override
  public void reset() {
  }

  @Override
  public void trimToSize(Encoder encoder) {

  }

  @Override
  public String describe() {
    return "BinaryTransform[" + getFeatureNamePrefix() + "]: threshold=" + threshold;
  }
}// END OF RealToBinaryTransform
