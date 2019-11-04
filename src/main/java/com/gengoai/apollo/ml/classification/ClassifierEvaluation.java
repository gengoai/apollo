/*
 * (c) 2005 David B. Bracewell
 *
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 *
 */

package com.gengoai.apollo.ml.classification;

import com.gengoai.apollo.ml.Example;
import com.gengoai.apollo.ml.FitParameters;
import com.gengoai.apollo.ml.Split;
import com.gengoai.apollo.ml.data.Dataset;
import com.gengoai.apollo.ml.vectorizer.IndexVectorizer;
import com.gengoai.apollo.ml.vectorizer.MultiLabelBinarizer;
import com.gengoai.logging.Logger;
import com.gengoai.stream.MStream;

import java.io.PrintStream;
import java.io.Serializable;

/**
 * <p>Defines common methods and metrics when evaluating classification models.</p>
 *
 * @author David B. Bracewell
 */
public abstract class ClassifierEvaluation implements Serializable {
   private static final Logger log = Logger.getLogger(ClassifierEvaluation.class);
   private static final long serialVersionUID = 1L;

   /**
    * Performs a cross-validation of the given classifier using the given dataset
    *
    * @param dataset       the dataset to perform cross-validation on
    * @param classifier    the classifier to train and test
    * @param fitParameters the {@link FitParameters} to use for training
    * @param nFolds        the number of folds to perform
    * @return the classifier evaluation
    */
   public static ClassifierEvaluation crossValidation(Dataset dataset,
                                                      Classifier classifier,
                                                      FitParameters<?> fitParameters,
                                                      int nFolds
                                                     ) {
      IndexVectorizer tmp = new MultiLabelBinarizer();
      tmp.fit(dataset);
      ClassifierEvaluation evaluation = tmp.size() <= 2
                                        ? new BinaryEvaluation(tmp)
                                        : new MultiClassEvaluation();
      int foldId = 0;
      for (Split split : dataset.shuffle().fold(nFolds)) {
         if (fitParameters.verbose.value()) {
            foldId++;
            log.info("Running fold {0}", foldId);
         }
         classifier.fit(split.train, fitParameters);
         evaluation.evaluate(classifier, split.test);
         if (fitParameters.verbose.value()) {
            log.info("Fold {0}: Cumulative Metrics(accuracy={1})", foldId, evaluation.accuracy());
         }
      }
      return evaluation;
   }

   /**
    * Evaluate the given {@link Classifier}. Will perform a binary classifier evaluation if the number of labels is two
    * or less and multi-class otherwise.
    *
    * @param classifier the classifier to evaluate
    * @param dataset    the dataset to evaluate on
    * @return the classifier evaluation
    */
   public static ClassifierEvaluation evaluateClassifier(Classifier classifier, Dataset dataset) {
      ClassifierEvaluation evaluation = classifier.getNumberOfLabels() <= 2
                                        ? new BinaryEvaluation(classifier.getPipeline().labelVectorizer)
                                        : new MultiClassEvaluation();
      evaluation.evaluate(classifier, dataset);
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
    * Evaluate the given model using the given dataset
    *
    * @param model   the model to evaluate
    * @param dataset the dataset to evaluate over
    */
   public final void evaluate(Classifier model, Dataset dataset) {
      evaluate(model, dataset.stream());
   }

   /**
    * Evaluate the given model using the given set of examples
    *
    * @param model   the model to evaluate
    * @param dataset the dataset to evaluate over
    */
   public abstract void evaluate(Classifier model, MStream<Example> dataset);

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
    * Merge this evaluation with another combining the results.
    *
    * @param evaluation the other evaluation to combine
    */
   public abstract void merge(ClassifierEvaluation evaluation);

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
    * Outputs the results of the classification to the given <code>PrintStream</code>
    *
    * @param printStream          the print stream to write to
    * @param printConfusionMatrix True print the confusion matrix, False do not print the confusion matrix.
    */
   public abstract void output(PrintStream printStream, boolean printConfusionMatrix);

   /**
    * Outputs the evaluation results to standard out.
    *
    * @param printConfusionMatrix True print the confusion matrix, False do not print the confusion matrix.
    */
   public void output(boolean printConfusionMatrix) {
      output(System.out, printConfusionMatrix);
   }

   /**
    * Outputs the evaluation results to standard out.
    */
   public void output() {
      output(System.out, false);
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
