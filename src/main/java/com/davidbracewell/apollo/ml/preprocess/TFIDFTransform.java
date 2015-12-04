package com.davidbracewell.apollo.ml.preprocess;

import com.davidbracewell.apollo.ml.Encoder;
import com.davidbracewell.apollo.ml.Feature;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.collection.Counter;
import com.davidbracewell.collection.Counters;
import com.google.common.util.concurrent.AtomicDouble;

import java.io.Serializable;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

/**
 * @author David B. Bracewell
 */
public class TFIDFTransform implements TransformProcessor<Instance>, Serializable {
  private final Pattern pattern;
  private volatile Counter<String> documentFrequencies = Counters.newConcurrentCounter();
  private volatile AtomicDouble totalDocs = new AtomicDouble();

  public TFIDFTransform(String featureNamePattern) {
    this.pattern = Pattern.compile(featureNamePattern);
  }

  @Override
  public void visit(Instance example) {
    totalDocs.addAndGet(1d);
    example.getFeatureSpace().distinct().filter(pattern.asPredicate()).forEach(documentFrequencies::increment);
  }

  @Override
  public Instance process(Instance example) {
    return Instance.create(
      example.stream().map(f -> {
          if (pattern.asPredicate().test(f.getName())) {
            return Feature.real(f.getName(),
              f.getValue() *
                Math.log(totalDocs.doubleValue() / (documentFrequencies.get(f.getName()) + 1)));
          }
          return f;
        }
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
