package com.davidbracewell.apollo.ml.classification.rule;

import com.davidbracewell.apollo.ml.Dataset;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.classification.Classifier;
import com.davidbracewell.apollo.ml.classification.ClassifierLearner;
import com.davidbracewell.conversion.Cast;
import lombok.NonNull;

/**
 * @author David B. Bracewell
 */
public class ZeroRLearner implements ClassifierLearner {

  @Override
  public Classifier train(@NonNull Dataset<Instance> dataset) {
    ZeroR model = new ZeroR(dataset.labelEncoder(), dataset.featureEncoder());
    model.distribution.merge(
      dataset.stream()
        .filter(Instance::hasLabel)
        .map(Instance::getLabel)
        .map(Cast::<String>as)
        .countByValue()
    );
    return model;
  }

}// END OF ZeroRLearner
