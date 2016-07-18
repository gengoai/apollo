package com.davidbracewell.apollo.ml.preprocess.transform;

import com.davidbracewell.apollo.ml.Feature;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.preprocess.RestrictedInstancePreprocessor;
import com.davidbracewell.collection.Counter;
import com.davidbracewell.collection.HashMapCounter;
import com.davidbracewell.stream.MStream;
import com.davidbracewell.stream.accumulator.MAccumulator;
import com.davidbracewell.string.StringUtils;

import java.io.Serializable;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * @author David B. Bracewell
 */
public class TFIDFTransform extends RestrictedInstancePreprocessor implements TransformProcessor<Instance>, Serializable {
  private static final long serialVersionUID = 1L;
  private volatile Counter<String> documentFrequencies = new HashMapCounter<>();
  private volatile double totalDocs = 0;
  private volatile AtomicBoolean finished = new AtomicBoolean(false);

  public TFIDFTransform() {
    super(StringUtils.EMPTY);
  }


  public TFIDFTransform(String featureNamePrefix) {
    super(featureNamePrefix);
  }


  @Override
  protected Stream<Feature> restrictedProcessImpl(Stream<Feature> featureStream, Instance originalExample) {
    double dSum = originalExample.getFeatures().size();
    return featureStream.map(f -> {
        double value = f.getValue() / dSum * Math.log(totalDocs / (documentFrequencies.get(f.getName()) + 1.0));
        if (value != 0) {
          return Feature.real(f.getName(), value);
        }
        return null;
      }
    )
      .filter(Objects::nonNull);
  }

  @Override
  protected void restrictedFitImpl(MStream<List<Feature>> stream) {
    MAccumulator<Double> docCount = stream.getContext().accumulator(0d);
    this.documentFrequencies.merge(
      stream.flatMap(instance -> {
          docCount.add(1d);
          return instance.stream().map(Feature::getName).distinct().collect(Collectors.toList());
        }
      ).countByValue()
    );
    this.totalDocs = docCount.value();
  }

  @Override
  public void reset() {
    totalDocs = 0;
    documentFrequencies.clear();
  }

  @Override
  public String describe() {
    if (acceptAll()) {
      return "TFIDFTransform";
    }
    return "TFIDFTransform[" + getRestriction() + "]";
  }


}// END OF TFIDFTransform
