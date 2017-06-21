package com.davidbracewell.apollo.ml.classification;

import com.davidbracewell.apollo.ml.BernoulliRBM;
import com.davidbracewell.apollo.ml.Feature;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.data.Dataset;
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
   public void reset() {

   }

   @Override
   protected RBMClassifier trainImpl(Dataset<Instance> dataset) {
      dataset.encode();

      int nV = dataset.getFeatureEncoder().size();
      BernoulliRBM rbm = new BernoulliRBM(nV, nHidden);
      rbm.train(dataset.asVectors().collect());

      boolean isMulticlass = dataset.getLabelEncoder().size() > 2;
      ClassifierLearner subLearner;
      if (isMulticlass) {
         subLearner = new BinarySGDLearner()
                         .oneVsRest();
      } else {
         subLearner = new BinarySGDLearner();
      }

      Dataset<Instance> transformed = Dataset.classification()
                                             .type(dataset.getType())
                                             .source(
                                                dataset.stream().map(i -> {
                                                   List<Feature> features = new ArrayList<>();
                                                   rbm.runVisible(i.toVector(dataset.getEncoderPair()))
                                                      .nonZeroIterator()
                                                      .forEachRemaining(
                                                         e -> features.add(
                                                            Feature.TRUE(Integer.toString(e.getIndex()))));
                                                   System.err.println(i + " : " + features);
                                                   return new Instance(features, i.getLabel());
                                                }));

      transformed.encode();
      RBMClassifier classifier = new RBMClassifier(dataset.getEncoderPair(), dataset.getPreprocessors());
      classifier.rbm = rbm;
      classifier.classifier = subLearner.train(transformed);
      return classifier;
   }


}// END OF RBMClassifierLearner
