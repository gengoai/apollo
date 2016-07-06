package com.davidbracewell.apollo.ml.preprocess.transform;

import com.davidbracewell.apollo.ml.Encoder;
import com.davidbracewell.apollo.ml.Feature;
import com.davidbracewell.collection.Counter;
import com.davidbracewell.collection.Counters;
import com.davidbracewell.collection.HashMapCounter;
import com.davidbracewell.string.StringUtils;
import com.google.common.util.concurrent.AtomicDouble;

import java.util.Collections;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.stream.Stream;

/**
 * @author David B. Bracewell
 */
public class TFIDFTransform extends RestrictedTransform {
  private static final long serialVersionUID = 1L;
  private volatile Counter<String> documentFrequencies = Counters.synchronizedCounter(new HashMapCounter<>());
  private volatile AtomicDouble totalDocs = new AtomicDouble();
  private volatile AtomicBoolean finished = new AtomicBoolean(false);

  public TFIDFTransform() {
    super(StringUtils.EMPTY);
  }


  public TFIDFTransform(String featureNamePrefix) {
    super(featureNamePrefix);
  }


  @Override
  protected void visitImpl(Stream<Feature> featureStream) {
    if (!finished.get()) {
      totalDocs.addAndGet(1d);
      featureStream.map(Feature::getName).distinct().forEach(documentFrequencies::increment);
    }
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
  public Set<String> finish(Set<String> removedFeature) {
    finished.set(true);
    return Collections.emptySet();
  }

  @Override
  public void reset() {
    finished.set(false);
    totalDocs.set(0);
    documentFrequencies.clear();
  }

  @Override
  public void trimToSize(Encoder encoder) {
    documentFrequencies = documentFrequencies.filterByKey(name -> encoder.encode(name) != -1);
  }

}// END OF TFIDFTransform
