package com.davidbracewell.apollo.ml.preprocess;

import com.davidbracewell.apollo.ml.Encoder;
import com.davidbracewell.apollo.ml.Feature;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.collection.Counter;
import com.davidbracewell.collection.Counters;
import com.google.common.util.concurrent.AtomicDouble;

import java.io.Serializable;
import java.util.stream.Collectors;

/**
 * @author David B. Bracewell
 */
public class TFIDFTransform implements TransformProcessor<Instance>, Serializable {
  private volatile Counter<String> documentFrequencies = Counters.newConcurrentCounter();
  private volatile AtomicDouble totalDocs = new AtomicDouble();

  @Override
  public void visit(Instance example) {
    totalDocs.addAndGet(1d);
    example.getFeatureSpace().distinct().forEach(documentFrequencies::increment);
  }

  @Override
  public Instance process(Instance example) {
    return Instance.create(
      example.stream().map(f ->
        Feature.real(f.getName(),
          f.getValue() *
            Math.log(totalDocs.doubleValue() / (documentFrequencies.get(f.getName()) + 1)))
      ).collect(Collectors.toList()),
      example.getLabel()
    );
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
