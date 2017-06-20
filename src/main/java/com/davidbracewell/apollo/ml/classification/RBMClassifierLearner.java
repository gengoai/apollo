package com.davidbracewell.apollo.ml.classification;

import com.davidbracewell.apollo.ml.BernoulliRBM;
import com.davidbracewell.apollo.ml.Feature;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.data.Dataset;
import com.davidbracewell.collection.map.Maps;
import lombok.Getter;
import lombok.Setter;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * @author David B. Bracewell
 */
public class RBMClassifierLearner extends ClassifierLearner {
   @Getter
   @Setter
   private int nHidden = 100;

   public static void main(String[] args) {
      Dataset<Instance> instances = Dataset.classification()
                                           .source(Arrays.asList(
                                              Instance.create(Maps.map("0", 1d, "1", 1d, "2", 1d), "A"),
                                              Instance.create(Maps.map("0", 1d, "2", 1d), "A"),
                                              Instance.create(Maps.map("0", 1d, "1", 1d, "2", 1d), "A"),
                                              Instance.create(Maps.map("3", 1d, "2", 1d, "4", 1d), "B"),
                                              Instance.create(Maps.map("4", 1d, "2", 1d), "B"),
                                              Instance.create(Maps.map("3", 1d, "4", 1d, "2", 1d), "B")
                                                                ));
      RBMClassifierLearner learner = new RBMClassifierLearner();
      learner.setNHidden(2);
      Classifier c = learner.train(instances);

      instances.forEach(i -> System.out.println(c.classify(i)));

   }

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
      System.err.println("isMultiClass=" + isMulticlass);
      ClassifierLearner subLearner;
      if (isMulticlass) {
         subLearner = new BinarySGDLearner()
                         .oneVsRest();
      } else {
         subLearner = new BinarySGDLearner();
//                         .setParameter("loss", new HingeLoss(0))
//                         .setParameter("activation", new StepActivation(0));
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
                                                   return new Instance(features);
                                                }));

      transformed.encode();
      RBMClassifier classifier = new RBMClassifier(dataset.getEncoderPair(), dataset.getPreprocessors());
      classifier.rbm = rbm;
      classifier.classifier = subLearner.train(transformed);
      return classifier;
   }


}// END OF RBMClassifierLearner
