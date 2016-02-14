package com.davidbracewell.apollo.ml.preprocess.transform;

import com.davidbracewell.apollo.ml.Encoder;
import com.davidbracewell.apollo.ml.Feature;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.collection.Counter;
import com.davidbracewell.collection.Counters;
import com.davidbracewell.string.StringUtils;
import com.google.common.util.concurrent.AtomicDouble;

import java.util.Objects;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * @author David B. Bracewell
 */
public class TFIDFTransform extends RestrictedTransform {
  private volatile Counter<String> documentFrequencies = Counters.newConcurrentCounter();
  private volatile AtomicDouble totalDocs = new AtomicDouble();

  public TFIDFTransform() {
    super(StringUtils.EMPTY);
  }


  public TFIDFTransform(String featureNamePrefix) {
    super(featureNamePrefix);
  }


  @Override
  protected void visitImpl(Stream<Feature> featureStream) {
    totalDocs.addAndGet(1d);
    featureStream.map(Feature::getName).distinct().forEach(documentFrequencies::increment);
  }


  @Override
  protected Stream<Feature> processImpl(Stream<Feature> featureStream) {
    return featureStream.map(f -> {
        double value = f.getValue() * Math.log((totalDocs.doubleValue() + 1) / (documentFrequencies.get(f.getName()) + 1));
        if (value != 0) {
          return Feature.real(f.getName(), value);
        }
        return null;
      }
    )
      .filter(Objects::nonNull);
  }

  @Override
  public void finish() {

  }

  @Override
  public void reset() {
    documentFrequencies.clear();
  }

  @Override
  public void trimToSize(Encoder encoder) {
    documentFrequencies = documentFrequencies.filterByKey(name -> encoder.encode(name) != -1);
  }

}// END OF TFIDFTransform
