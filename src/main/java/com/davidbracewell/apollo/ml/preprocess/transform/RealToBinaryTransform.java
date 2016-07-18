package com.davidbracewell.apollo.ml.preprocess.transform;

import com.davidbracewell.apollo.ml.Feature;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.preprocess.RestrictedInstancePreprocessor;
import com.davidbracewell.stream.MStream;
import com.davidbracewell.string.StringUtils;
import lombok.NonNull;

import java.io.Serializable;
import java.util.List;
import java.util.stream.Stream;

/**
 * @author David B. Bracewell
 */
public class RealToBinaryTransform extends RestrictedInstancePreprocessor implements TransformProcessor<Instance>, Serializable {
  private static final long serialVersionUID = 1L;
  private final double threshold;

  public RealToBinaryTransform(double threshold) {
    this(StringUtils.EMPTY, threshold);
  }

  public RealToBinaryTransform(@NonNull String featureNamePrefix, double threshold) {
    super(featureNamePrefix);
    this.threshold = threshold;
  }


  @Override
  public void reset() {
  }

  @Override
  public String describe() {
    if (acceptAll()) {
      return "RealToBinaryTransform: threshold=" + threshold;
    }
    return "RealToBinaryTransform[" + getRestriction() + "]: threshold=" + threshold;
  }

  @Override
  protected void restrictedFitImpl(MStream<List<Feature>> stream) {

  }

  @Override
  public boolean requiresFit() {
    return false;
  }

  @Override
  protected Stream<Feature> restrictedProcessImpl(Stream<Feature> featureStream, Instance originalExample) {
    return featureStream.filter(f -> f.getValue() >= threshold).map(feature -> Feature.TRUE(feature.getName()));
  }

}// END OF RealToBinaryTransform
