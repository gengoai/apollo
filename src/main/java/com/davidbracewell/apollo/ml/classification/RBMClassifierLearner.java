package com.davidbracewell.apollo.ml.classification;

import com.davidbracewell.apollo.ml.Feature;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.data.Dataset;
import com.davidbracewell.apollo.ml.nn.BernoulliRBM;
import lombok.Getter;
import lombok.Setter;

import java.util.ArrayList;
import java.util.List;

/**
 * @author David B. Bracewell
 */
public class RBMClassifierLearner extends ClassifierLearner {
   @Getter
   @Setter
   private int nHidden = 100;

   @Override
   public void resetLearnerParameters() {

   }

   @Override
   protected RBMClassifier trainImpl(Dataset<Instance> dataset) {
      BernoulliRBM rbm = new BernoulliRBM(dataset.getFeatureEncoder().size(), nHidden);
      rbm.train(dataset.stream().map(ii -> ii.toVector(dataset.getEncoderPair())).collect());
      LibLinearLearner subLearner = new LibLinearLearner();
      Dataset<Instance> transformed = Dataset.classification()
                                             .type(dataset.getType())
                                             .source(
                                                dataset.stream().map(i -> {
                                                   List<Feature> features = new ArrayList<>();
                                                   rbm.runVisibleProbs(i.toVector(dataset.getEncoderPair()))
                                                      .nonZeroIterator()
                                                      .forEachRemaining(
                                                         e -> features.add(
                                                            Feature.TRUE(Integer.toString(e.getIndex()))));
                                                   return new Instance(features, i.getLabel());
                                                }));
      RBMClassifier classifier = new RBMClassifier(this);
      classifier.rbm = rbm;
      classifier.classifier = subLearner.train(transformed);
      return classifier;
   }


}// END OF RBMClassifierLearner
