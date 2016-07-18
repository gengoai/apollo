package com.davidbracewell.apollo.ml.preprocess.transform;

import com.davidbracewell.apollo.ml.Feature;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.preprocess.RestrictedInstancePreprocessor;
import com.davidbracewell.collection.EnhancedDoubleStatistics;
import com.davidbracewell.stream.MStream;
import com.davidbracewell.stream.accumulator.MAccumulator;
import com.davidbracewell.stream.accumulator.StatisticsAccumulatable;
import com.davidbracewell.string.StringUtils;

import java.io.Serializable;
import java.util.List;
import java.util.stream.Stream;

/**
 * @author David B. Bracewell
 */
public class ZScoreTransform extends RestrictedInstancePreprocessor implements TransformProcessor<Instance>, Serializable {

  private static final long serialVersionUID = 1L;
  private final EnhancedDoubleStatistics statistics = new EnhancedDoubleStatistics();

  public ZScoreTransform() {
    super(StringUtils.EMPTY);
  }

  public ZScoreTransform(String featureNamePrefix) {
    super(featureNamePrefix);
  }

  @Override
  protected Stream<Feature> restrictedProcessImpl(Stream<Feature> featureStream, Instance originalExample) {
    return featureStream.map(feature -> Feature.real(feature.getName(), (feature.getValue() - statistics.getAverage()) / statistics.getSampleStandardDeviation()));
  }


  @Override
  protected void restrictedFitImpl(MStream<List<Feature>> stream) {
    MAccumulator<EnhancedDoubleStatistics> stats = stream.getContext().accumulator(null, new StatisticsAccumulatable());
    stream.forEach(instance ->
      stats.add(
        instance.stream().mapToDouble(Feature::getValue).collect(
          EnhancedDoubleStatistics::new, EnhancedDoubleStatistics::accept, EnhancedDoubleStatistics::combine
        )
      )
    );
    this.statistics.combine(stats.value());
  }

  @Override
  public void reset() {
    statistics.clear();
  }

  @Override
  public String describe() {
    if (acceptAll()) {
      return "ZScoreTransform";
    }
    return "ZScoreTransform[" + getRestriction() + "]";
  }

}// END OF ZScoreTransform
