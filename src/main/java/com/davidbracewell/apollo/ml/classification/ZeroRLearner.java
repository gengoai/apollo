package com.davidbracewell.apollo.ml.classification;

import com.davidbracewell.apollo.ml.Dataset;
import com.davidbracewell.apollo.ml.Instance;
import lombok.NonNull;

import java.util.Map;

/**
 * @author David B. Bracewell
 */
public class ZeroRLearner extends ClassifierLearner {
  private static final long serialVersionUID = 1L;

  @Override
  public Classifier trainImpl(@NonNull Dataset<Instance> dataset) {
    ZeroR model = new ZeroR(dataset.getEncoderPair());
    Map<String, Long> m = dataset.stream()
      .filter(Instance::hasLabel)
      .map(Instance::getLabel)
      .map(Object::toString)
      .countByValue();

    model.distribution = new double[model.numberOfLabels()];
    m.forEach((label, value) -> model.distribution[(int) model.getLabelEncoder().encode(label)] = value);
    return model;
  }

  @Override
  public void reset() {

  }

}// END OF ZeroRLearner
