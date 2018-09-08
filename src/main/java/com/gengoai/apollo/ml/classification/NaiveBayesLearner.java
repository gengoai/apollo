package com.gengoai.apollo.ml.classification;

import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.ml.Instance;
import com.gengoai.apollo.ml.data.Dataset;
import com.gengoai.collection.Iterables;
import lombok.Getter;
import lombok.NonNull;
import lombok.Setter;

import java.util.Arrays;

/**
 * <p>Trains three variations of Naive Bayes models specifically suited for text classification.</p>
 *
 * @author David B. Bracewell
 */
public class NaiveBayesLearner extends ClassifierLearner {
   private static final long serialVersionUID = 1L;
   @Getter
   @Setter
   private volatile NaiveBayes.ModelType modelType;

   /**
    * Instantiates a new Naive bayes learner.
    */
   public NaiveBayesLearner() {
      this(NaiveBayes.ModelType.Bernoulli);
   }

   /**
    * Instantiates a new Naive bayes learner.
    *
    * @param modelType the model type
    */
   public NaiveBayesLearner(@NonNull NaiveBayes.ModelType modelType) {
      this.modelType = modelType;
   }

   @Override
   public void resetLearnerParameters() {

   }

   @Override
   protected NaiveBayes trainImpl(Dataset<Instance> dataset) {
      NaiveBayes model = new NaiveBayes(this, modelType);
      model.conditionals = new double[model.numberOfFeatures()][model.numberOfLabels()];
      model.priors = new double[model.numberOfLabels()];
      double[] labelCounts = new double[model.numberOfLabels()];

      double N = 0;
      for (Instance instance : dataset) {
         if (instance.hasLabel()) {
            N++;
            int ci = (int) model.encodeLabel(instance.getLabel());
            model.priors[ci] += instance.getWeight();
            NDArray vector = instance.toVector(dataset.getEncoderPair());
            for (NDArray.Entry entry : Iterables.asIterable(vector.sparseIterator())) {
               labelCounts[ci] += entry.getValue();
               model.conditionals[(int) entry.getIndex()][ci] += instance.getWeight() * modelType.convertValue(
                  entry.getValue());
            }
         }
      }

      double V = model.numberOfFeatures();
      for (int featureIndex = 0; featureIndex < model.conditionals.length; featureIndex++) {
         double[] tmp = Arrays.copyOf(model.conditionals[featureIndex], model.conditionals[featureIndex].length);
         for (int labelIndex = 0; labelIndex < model.priors.length; labelIndex++) {
            if (modelType == NaiveBayes.ModelType.Complementary) {
               double nCi = 0;
               double nC = 0;
               for (int j = 0; j < model.priors.length; j++) {
                  if (j != labelIndex) {
                     nCi += tmp[j];
                     nC += labelCounts[j];
                  }
               }
               model.conditionals[featureIndex][labelIndex] = Math.log(
                  modelType.normalize(nCi, model.priors[labelIndex], nC, V));
            } else {
               model.conditionals[featureIndex][labelIndex] = Math.log(
                  modelType.normalize(model.conditionals[featureIndex][labelIndex], model.priors[labelIndex],
                                      labelCounts[labelIndex], V));
            }
         }

      }

      for (int i = 0; i < model.priors.length; i++) {
         model.priors[i] = Math.log(model.priors[i] / N);
      }

      return model;
   }

}// END OF NaiveBayesLearner
