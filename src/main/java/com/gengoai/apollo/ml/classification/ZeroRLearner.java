package com.gengoai.apollo.ml.classification;

import com.gengoai.apollo.ml.Instance;
import com.gengoai.apollo.ml.data.Dataset;
import com.gengoai.apollo.ml.Instance;
import com.gengoai.apollo.ml.data.Dataset;
import lombok.NonNull;

import java.util.Map;

/**
 * Learner for ZeroR classifiers which always predict the majority class.
 *
 * @author David B. Bracewell
 */
public class ZeroRLearner extends ClassifierLearner {
   private static final long serialVersionUID = 1L;

   @Override
   public Classifier trainImpl(@NonNull Dataset<Instance> dataset) {
      ZeroR model = new ZeroR(this);
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
   public void resetLearnerParameters() {

   }

}// END OF ZeroRLearner
