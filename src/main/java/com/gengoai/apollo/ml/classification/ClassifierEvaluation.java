package com.gengoai.apollo.ml.classification;

import com.gengoai.apollo.ml.Evaluation;
import com.gengoai.apollo.ml.FitParameters;
import com.gengoai.apollo.ml.Split;
import com.gengoai.apollo.ml.data.Dataset;
import com.gengoai.apollo.ml.vectorizer.BinaryLabelVectorizer;
import com.gengoai.conversion.Cast;
import com.gengoai.logging.Logger;

import java.io.Serializable;
import java.util.function.Consumer;

/**
 * <p>Defines common methods and metrics when evaluating classification models.</p>
 *
 * @author David B. Bracewell
 */
public abstract class ClassifierEvaluation implements Evaluation<Classifier>, Serializable {
   private static final Logger log = Logger.getLogger(ClassifierEvaluation.class);
   private static final long serialVersionUID = 1L;


   public static ClassifierEvaluation evaluateClassifier(Classifier classifier, Dataset dataset) {
      ClassifierEvaluation evaluation = classifier.getNumberOfLabels() <= 2
                                        ? new BinaryEvaluation(classifier.getLabelVectorizer())
                                        : new MultiClassEvaluation();
      evaluation.evaluate(classifier, dataset);
      return evaluation;
   }

   /**
    * Cross validation multi class evaluation.
    *
    * @param dataset    the dataset
    * @param classifier the classifier
    * @param updater    the updater
    * @param nFolds     the n folds
    * @return the multi class evaluation
    */
   public static ClassifierEvaluation crossValidation(Dataset dataset,
                                                      Classifier classifier,
                                                      Consumer<? extends FitParameters> updater,
                                                      int nFolds
                                                     ) {
      FitParameters parameters = classifier.getDefaultFitParameters();
      updater.accept(Cast.as(parameters));
      return crossValidation(dataset, classifier, parameters, nFolds);
   }

   /**
    * Cross validation multi class evaluation.
    *
    * @param dataset       the dataset
    * @param classifier    the classifier
    * @param fitParameters the fit parameters
    * @param nFolds        the n folds
    * @return the multi class evaluation
    */
   public static ClassifierEvaluation crossValidation(Dataset dataset,
                                                      Classifier classifier,
                                                      FitParameters fitParameters,
                                                      int nFolds
                                                     ) {

      ClassifierEvaluation evaluation = classifier.getLabelVectorizer() instanceof BinaryLabelVectorizer
                                        ? new BinaryEvaluation(classifier.getLabelVectorizer())
                                        : new MultiClassEvaluation();
      int foldId = 0;
      for (Split split : dataset.shuffle().fold(nFolds)) {
         if (fitParameters.verbose) {
            foldId++;
            log.info("Running fold {0}", foldId);
         }
         classifier.fit(split.train, fitParameters);
         evaluation.evaluate(classifier, split.test);
         if (fitParameters.verbose) {
            log.info("Fold {0}: Cumulative Metrics(accuracy={1})", foldId, evaluation.accuracy());
         }
      }
      return evaluation;
   }


   /**
    * <p>Calculates the accuracy, which is the percentage of items correctly classified.</p>
    *
    * @return the accuracy
    */
   public abstract double accuracy();

   /**
    * <p>Calculate the diagnostic odds ratio which is <code> positive likelihood ration / negative likelihood
    * ratio</code>. The diagnostic odds ratio is taken from the medical field and measures the effectiveness of a
    * medical tests. The measure works for binary classifications and provides the odds of being classified true when
    * the correct classification is false.</p>
    *
    * @return the diagnostic odds ratio
    */
   public double diagnosticOddsRatio() {
      return positiveLikelihoodRatio() / negativeLikelihoodRatio();
   }

   /**
    * Adds a prediction entry to the evaluation.
    *
    * @param gold      the gold, or actual, label
    * @param predicted the model predicted label
    */
   public abstract void entry(String gold, Classification predicted);


   /**
    * Calculates the false negative rate, which is calculated as <code>False Positives / (True Positives + False
    * Positives)</code>
    *
    * @return the false negative rate
    */
   public double falseNegativeRate() {
      double fn = falseNegatives();
      double tp = truePositives();
      if (tp + fn == 0) {
         return 0.0;
      }
      return fn / (fn + tp);
   }

   /**
    * Calculates the number of false negatives
    *
    * @return the number of false negatives
    */
   public abstract double falseNegatives();

   /**
    * Calculates the false omission rate (or Negative Predictive Value), which is calculated as <code>False Negatives /
    * (False Negatives + True Negatives)</code>
    *
    * @return the false omission rate
    */
   public double falseOmissionRate() {
      double fn = falseNegatives();
      double tn = trueNegatives();
      if (tn + fn == 0) {
         return 0.0;
      }
      return fn / (fn + tn);
   }

   /**
    * Calculates the false positive rate which is calculated as <code>False Positives / (True Negatives + False
    * Positives)</code>
    *
    * @return the false positive rate
    */
   public double falsePositiveRate() {
      double tn = trueNegatives();
      double fp = falsePositives();
      if (tn + fp == 0) {
         return 0.0;
      }
      return fp / (tn + fp);
   }

   /**
    * Calculates the number of false positives
    *
    * @return the number of false positives
    */
   public abstract double falsePositives();

   /**
    * Calculates the negative likelihood ratio, which is <code>False Positive Rate / Specificity</code>
    *
    * @return the negative likelihood ratio
    */
   public double negativeLikelihoodRatio() {
      return falseNegativeRate() / specificity();
   }

   /**
    * Proportion of negative results that are true negative.
    *
    * @return the double
    */
   public double negativePredictiveValue() {
      double tn = trueNegatives();
      double fn = falseNegatives();
      if (tn + fn == 0) {
         return 0;
      }
      return tn / (tn + fn);
   }

   /**
    * Calculates the positive likelihood ratio, which is <code>True Positive Rate / False Positive Rate</code>
    *
    * @return the positive likelihood ratio
    */
   public double positiveLikelihoodRatio() {
      return truePositiveRate() / falsePositiveRate();
   }

   /**
    * Calculates the sensitivity (same as the micro-averaged recall)
    *
    * @return the sensitivity
    */
   public double sensitivity() {
      double tp = truePositives();
      double fn = falseNegatives();
      if (tp + fn == 0) {
         return 0.0;
      }
      return tp / (tp + fn);
   }

   /**
    * Calculates the specificity, which is <code>True Negatives / (True Negatives + False Positives)</code>
    *
    * @return the specificity
    */
   public double specificity() {
      double tn = trueNegatives();
      double fp = falsePositives();
      if (tn + fp == 0) {
         return 1.0;
      }
      return tn / (tn + fp);
   }

   /**
    * Calculates the true negative rate (or specificity)
    *
    * @return the true negative rate
    */
   public double trueNegativeRate() {
      return specificity();
   }

   /**
    * Counts the number of true negatives
    *
    * @return the number of true negatives
    */
   public abstract double trueNegatives();

   /**
    * Calculates the true positive rate (same as micro recall).
    *
    * @return the true positive rate
    */
   public double truePositiveRate() {
      return sensitivity();
   }

   /**
    * Calculates the number of true positives.
    *
    * @return the number of true positive
    */
   public abstract double truePositives();


}//END OF ClassifierEvaluation
