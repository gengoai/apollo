package com.gengoai.apollo.ml.classification;

import com.gengoai.apollo.ml.Evaluation;
import com.gengoai.apollo.ml.Instance;
import com.gengoai.apollo.ml.data.Dataset;
import lombok.NonNull;

import java.io.Serializable;

/**
 * @author David B. Bracewell
 */
public interface ClassifierEvaluation extends Evaluation<Instance, Classifier>, Serializable {

   /**
    * Evaluate model classifier evaluation.
    *
    * @param classifier the classifier
    * @param testSet    the test set
    * @return the classifier evaluation
    */
   static ClassifierEvaluation evaluateModel(@NonNull Classifier classifier, @NonNull Dataset<Instance> testSet) {
      ClassifierEvaluation evaluation = null;
      if (classifier.getLabelEncoder().size() == 2) {
         evaluation = new BinaryEvaluation(classifier.getLabelEncoder());
      } else {
         evaluation = new MultiClassEvaluation();
      }
      evaluation.evaluate(classifier, testSet);
      return evaluation;
   }

   /**
    * <p>Calculates the accuracy, which is the percentage of items correctly classified.</p>
    *
    * @return the accuracy
    */
   double accuracy();

   /**
    * Calculates the false negative rate, which is calculated as <code>False Positives / (True Positives + False
    * Positives)</code>
    *
    * @return the false negative rate
    */
   default double falseNegativeRate() {
      double tp = truePositives();
      double fn = falseNegatives();
      if (tp + fn == 0) {
         return 0.0;
      }
      return fn / (tp + fn);
   }

   /**
    * Calculates the number of false negatives
    *
    * @return the number of false negatives
    */
   double falseNegatives();

   /**
    * Calculates the false omission rate (or Negative Predictive Value), which is calculated as <code>False Negatives /
    * (False Negatives + True Negatives)</code>
    *
    * @return the false omission rate
    */
   default double falseOmissionRate() {
      double fn = falseNegatives();
      double tn = trueNegatives();
      if (tn + fn == 0) {
         return 0.0;
      }
      return fn / (fn + tn);
   }

   /**
    * Calculates the number of false positives
    *
    * @return the number of false positives
    */
   double falsePositives();

   /**
    * Calculates the false positive rate which is calculated as <code>False Positives / (True Negatives + False
    * Positives)</code>
    *
    * @return the false positive rate
    */
   default double falsePositiveRate() {
      double tn = trueNegatives();
      double fp = falsePositives();
      if (tn + fp == 0) {
         return 0.0;
      }
      return fp / (tn + fp);
   }

   /**
    * Calculates the sensitivity (same as the micro-averaged recall)
    *
    * @return the sensitivity
    */
   default double sensitivity() {
      double tp = truePositives();
      double fn = falseNegatives();
      if (tp + fn == 0) {
         return 0.0;
      }
      return tp / (tp + fn);
   }

   /**
    * Calculates the number of true positives.
    *
    * @return the number of true positive
    */
   double truePositives();


   /**
    * Calculates the true positive rate (same as micro recall).
    *
    * @return the true positive rate
    */
   default double truePositiveRate() {
      return sensitivity();
   }

   /**
    * Counts the number of true negatives
    *
    * @return the number of true negatives
    */
   double trueNegatives();

   /**
    * Calculates the true negative rate (or specificity)
    *
    * @return the true negative rate
    */
   default double trueNegativeRate() {
      return specificity();
   }


   /**
    * Calculates the specificity, which is <code>True Negatives / (True Negatives + False Positives)</code>
    *
    * @return the specificity
    */
   default double specificity() {
      double tn = trueNegatives();
      double fp = falsePositives();
      if (tn + fp == 0) {
         return 1.0;
      }
      return tn / (tn + fp);
   }


   /**
    * <p>Calculate the diagnostic odds ratio which is <code> positive likelihood ration / negative likelihood
    * ratio</code>. The diagnostic odds ratio is taken from the medical field and measures the effectiveness of a
    * medical tests. The measure works for binary classifications and provides the odds of being classified true when
    * the correct classification is false.</p>
    *
    * @return the diagnostic odds ratio
    */
   default double diagnosticOddsRatio() {
      return positiveLikelihoodRatio() / negativeLikelihoodRatio();
   }

   /**
    * Calculates the positive likelihood ratio, which is <code>True Positive Rate / False Positive Rate</code>
    *
    * @return the positive likelihood ratio
    */
   default double positiveLikelihoodRatio() {
      return truePositiveRate() / falsePositiveRate();
   }


   /**
    * Calculates the negative likelihood ratio, which is <code>False Positive Rate / Specificity</code>
    *
    * @return the negative likelihood ratio
    */
   default double negativeLikelihoodRatio() {
      return falseNegativeRate() / specificity();
   }


}//END OF ClassifierEvaluation