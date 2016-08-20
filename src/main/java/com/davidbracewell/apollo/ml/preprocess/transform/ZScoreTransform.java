package com.davidbracewell.apollo.ml.preprocess.transform;

import com.davidbracewell.EnhancedDoubleStatistics;
import com.davidbracewell.apollo.ml.Feature;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.preprocess.RestrictedInstancePreprocessor;
import com.davidbracewell.io.structured.ElementType;
import com.davidbracewell.io.structured.StructuredReader;
import com.davidbracewell.io.structured.StructuredWriter;
import com.davidbracewell.stream.MStream;
import com.davidbracewell.stream.accumulator.MAccumulator;
import com.davidbracewell.stream.accumulator.StatisticsAccumulatable;
import com.davidbracewell.string.StringUtils;
import lombok.NonNull;

import java.io.IOException;
import java.io.Serializable;
import java.util.List;
import java.util.stream.Stream;

/**
 * @author David B. Bracewell
 */
public class ZScoreTransform extends RestrictedInstancePreprocessor implements TransformProcessor<Instance>, Serializable {

  private static final long serialVersionUID = 1L;
  private double mean = 0;
  private double standardDeviation = 0;

  public ZScoreTransform() {
    super(StringUtils.EMPTY);
  }

  public ZScoreTransform(String featureNamePrefix) {
    super(featureNamePrefix);
  }

  @Override
  protected Stream<Feature> restrictedProcessImpl(Stream<Feature> featureStream, Instance originalExample) {
    return featureStream.map(feature -> Feature.real(feature.getName(),
                                                     (feature.getValue() - mean) / standardDeviation));
  }


  @Override
  protected void restrictedFitImpl(MStream<List<Feature>> stream) {
    MAccumulator<EnhancedDoubleStatistics> stats = stream.getContext().accumulator(null, new StatisticsAccumulatable());
    stream.forEach(instance ->
                     stats.add(
                       instance.stream().mapToDouble(Feature::getValue).collect(
                         EnhancedDoubleStatistics::new,
                         EnhancedDoubleStatistics::accept,
                         EnhancedDoubleStatistics::combine
                       )
                     )
    );
    this.mean = stats.value().getAverage();
    this.standardDeviation = stats.value().getSampleStandardDeviation();
  }

  @Override
  public void reset() {
    this.mean = 0;
    this.standardDeviation = 0;
  }

  @Override
  public String describe() {
    if (acceptAll()) {
      return "ZScoreTransform{mean=" + mean + ", std=" + standardDeviation + "}";
    }
    return "ZScoreTransform[" + getRestriction() + "]{mean=" + mean + ", std=" + standardDeviation + "}";
  }

  @Override
  public void write(@NonNull StructuredWriter writer) throws IOException {
    if (!acceptAll()) {
      writer.writeKeyValue("restriction", getRestriction());
    }
    writer.writeKeyValue("mean", mean);
    writer.writeKeyValue("stddev", standardDeviation);
  }

  @Override
  public void read(@NonNull StructuredReader reader) throws IOException {
    reset();
    while (reader.peek() != ElementType.END_OBJECT) {
      switch (reader.peekName()) {
        case "restriction":
          setRestriction(reader.nextKeyValue().v2.asString());
          break;
        case "mean":
          this.mean = reader.nextKeyValue().v2.asDoubleValue();
          break;
        case "stddev":
          this.standardDeviation = reader.nextKeyValue().v2.asDoubleValue();
          break;
      }
    }
  }

}// END OF ZScoreTransform
