package com.davidbracewell.apollo.ml.preprocess.transform;

import com.davidbracewell.apollo.ml.Feature;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.preprocess.RestrictedInstancePreprocessor;
import com.davidbracewell.collection.EnhancedDoubleStatistics;
import com.davidbracewell.stream.MStream;
import com.davidbracewell.stream.accumulator.MAccumulator;
import com.davidbracewell.stream.accumulator.StatisticsAccumulatable;
import com.google.common.base.Preconditions;
import lombok.NonNull;

import java.io.Serializable;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Stream;

/**
 * @author David B. Bracewell
 */
public class RealToDiscreteTransform extends RestrictedInstancePreprocessor implements TransformProcessor<Instance>, Serializable {
  private static final long serialVersionUID = 1L;
  private final double[] bins;


  public RealToDiscreteTransform(int numberOfBins) {
    super(null);
    Preconditions.checkArgument(numberOfBins > 0, "Number of bins must be > 0.");
    this.bins = new double[numberOfBins];
  }

  public RealToDiscreteTransform(@NonNull String featureNamePrefix, int numberOfBins) {
    super(featureNamePrefix);
    Preconditions.checkArgument(numberOfBins > 0, "Number of bins must be > 0.");
    this.bins = new double[numberOfBins];
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
    EnhancedDoubleStatistics statistics = stats.value();
    double max = statistics.getMax();
    double min = statistics.getMin();
    double binSize = ((max - min) / bins.length);
    double sum = 0;
    for (int i = 0; i < bins.length; i++) {
      sum += binSize;
      bins[i] = sum;
    }
  }

  @Override
  public void reset() {
    Arrays.fill(bins, 0);
  }

  @Override
  protected Stream<Feature> restrictedProcessImpl(Stream<Feature> featureStream, Instance originalExample) {
    return featureStream.map(f -> {
      for (int i = 0; i < bins.length; i++) {
        if (f.getValue() < bins[i]) {
          return Feature.TRUE(f.getName(), Integer.toString(i));
        }
      }
      return Feature.TRUE(f.getName(), Integer.toString(bins.length - 1));
    });
  }


  @Override
  public String describe() {
    if (acceptAll()) {
      return "RealToDiscreteTransform: numberOfBins=" + bins.length;
    }
    return "RealToDiscreteTransform[" + getRestriction() + "]: numberOfBins=" + bins.length;
  }


}// END OF RealToDiscreteTransform
