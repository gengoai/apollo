package com.davidbracewell.apollo.ml.preprocess.transform;

import com.davidbracewell.apollo.ml.Encoder;
import com.davidbracewell.apollo.ml.Feature;
import com.davidbracewell.collection.Counter;
import com.davidbracewell.collection.Counters;

import java.util.concurrent.atomic.AtomicBoolean;
import java.util.stream.Stream;

/**
 * @author David B. Bracewell
 */
public class RealToDiscreteTransform extends RestrictedTransform {
  private static final long serialVersionUID = 1L;
  private final double[] bins;
  private transient Counter<Double> counts = Counters.newHashMapCounter();
  private volatile AtomicBoolean finished = new AtomicBoolean(false);


  protected RealToDiscreteTransform(String featureNamePrefix, int numberOfBins) {
    super(featureNamePrefix);
    this.bins = new double[numberOfBins];
  }

  @Override
  protected void visitImpl(Stream<Feature> featureStream) {
    if (!finished.get()) {
      featureStream.mapToDouble(Feature::getValue).forEach(counts::increment);
    }
  }

  @Override
  protected Stream<Feature> processImpl(Stream<Feature> featureStream) {
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
  public void finish() {
    double max = counts.max();
    double min = counts.min();
    double binSize = ((max - min) / bins.length);
    double sum = 0;
    for (int i = 0; i < bins.length; i++) {
      sum += binSize;
      bins[i] = sum;
    }
    finished.set(false);
    counts.clear();
  }

  @Override
  public void reset() {

  }

  @Override
  public void trimToSize(Encoder encoder) {

  }
}// END OF RealToDiscreteTransform
