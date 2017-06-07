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
 */

package com.davidbracewell.apollo.ml.classification;

import com.davidbracewell.apollo.ml.Evaluation;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.data.Dataset;
import com.davidbracewell.collection.counter.Counter;
import com.davidbracewell.collection.counter.Counters;
import com.davidbracewell.collection.counter.HashMapMultiCounter;
import com.davidbracewell.collection.counter.MultiCounter;
import com.davidbracewell.conversion.Cast;
import com.davidbracewell.guava.common.base.Preconditions;
import com.davidbracewell.logging.Logger;
import com.davidbracewell.string.StringUtils;
import com.davidbracewell.string.TableFormatter;
import com.davidbracewell.tuple.Tuple2;
import lombok.NonNull;

import java.io.PrintStream;
import java.io.Serializable;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Supplier;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * Provides various common metrics for measuring the quality of classifiers.
 *
 * @author David B. Bracewell
 */
public class ClassifierEvaluation implements Evaluation<Instance, Classifier>, Serializable {
   private static final Logger log = Logger.getLogger(ClassifierEvaluation.class);
   private static final long serialVersionUID = 1L;
   private final MultiCounter<String, String> matrix = new HashMapMultiCounter<>();
   private double total = 0;

   /**
    * Cross validation classifier evaluation.
    *
    * @param dataset         the dataset
    * @param learnerSupplier the learner supplier
    * @param nFolds          the n folds
    * @return the classifier evaluation
    */
   public static ClassifierEvaluation crossValidation(@NonNull Dataset<Instance> dataset, @NonNull Supplier<ClassifierLearner> learnerSupplier, int nFolds) {
      ClassifierEvaluation evaluation = new ClassifierEvaluation();
      AtomicInteger foldId = new AtomicInteger(0);
      dataset.fold(nFolds).forEach((train, test) -> {
         log.info("Running fold {0}", foldId.incrementAndGet());
         Classifier model = learnerSupplier.get().train(train);
         evaluation.evaluate(model, test);
         log.info("Fold {0}: Cumulative Metrics(microP={1}, microR={2}, microF1={3})", foldId.get(),
                  evaluation.microPrecision(),
                  evaluation.microRecall(),
                  evaluation.microF1());
      });
      return evaluation;
   }

   /**
    * Evaluate model classifier evaluation.
    *
    * @param classifier the classifier
    * @param testSet    the test set
    * @return the classifier evaluation
    */
   public static ClassifierEvaluation evaluateModel(@NonNull Classifier classifier, @NonNull Dataset<Instance> testSet) {
      ClassifierEvaluation evaluation = new ClassifierEvaluation();
      evaluation.evaluate(classifier, testSet);
      return evaluation;
   }

   /**
    * <p>Calculates the accuracy, which is the percentage of items correctly classified.</p>
    *
    * @return the accuracy
    */
   public double accuracy() {
      double correct = matrix.firstKeys().stream()
                             .mapToDouble(k -> matrix.get(k, k))
                             .sum();
      return correct / total;
   }

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
    * Calculate the diagnostic odds ratio for the given label
    *
    * @param label the label to calculate the diagnostic odds ratio for.
    * @return the diagnostic odds ratio
    */
   public double diagnosticOddsRatio(String label) {
      return positiveLikelihoodRatio(label) / negativeLikelihoodRatio(label);
   }

   /**
    * Adds a prediction entry to the evaluation.
    *
    * @param gold      the gold, or actual, label
    * @param predicted the model predicted label
    */
   public void entry(String gold, String predicted) {
      matrix.increment(gold, predicted);
      total++;
   }

   @Override
   public void evaluate(@NonNull Classifier model, @NonNull Dataset<Instance> dataset) {
      dataset.stream()
             .filter(Instance::hasLabel)
             .mapToPair(instance -> Tuple2.of(instance.getLabel().toString(), model.classify(instance).getResult()))
             .forEachLocal(this::entry);
   }

   @Override
   public void evaluate(@NonNull Classifier model, @NonNull Collection<Instance> dataset) {
      dataset.forEach(instance ->
                         entry(
                            instance.getLabel().toString(),
                            model.classify(instance).getResult()
                              )
                     );
   }

   private double f1(double p, double r) {
      if (p + r == 0) {
         return 0;
      }
      return (2 * p * r) / (p + r);
   }

   /**
    * Calculates the F1-measure for the given label, which is calculated as <code>(2 * precision(label) * recall(label))
    * / (precision(label) + recall(label)</code>
    *
    * @param label the label to calculate the F1 measure for
    * @return the f1 measure
    */
   public double f1(String label) {
      return f1(precision(label), recall(label));
   }

   /**
    * Calculates the F1 measure for each class
    *
    * @return a Counter where the items are labels and the values are F1 scores
    */
   public Counter<String> f1PerClass() {
      Counter<String> f1 = Counters.newCounter();
      Counter<String> p = precisionPerClass();
      Counter<String> r = recallPerClass();
      matrix.firstKeys().forEach(k -> f1.set(k, f1(p.get(k), r.get(k))));
      return f1;
   }

   /**
    * Calculates the false negative rate of the given label
    *
    * @param label the label to calculate the false negative rate of
    * @return the false negative rate
    */
   public double falseNegativeRate(String label) {
      double tp = truePositives(label);
      double fn = falseNegatives(label);
      if (tp + fn == 0) {
         return 0.0;
      }
      return fn / (tp + fn);
   }

   /**
    * Calculates the false negative rate, which is calculated as <code>False Positives / (True Positives + False
    * Positives)</code>
    *
    * @return the false negative rate
    */
   public double falseNegativeRate() {
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
   public double falseNegatives() {
      return matrix.firstKeys().stream().mapToDouble(k -> matrix.get(k).sum() - matrix.get(k, k)).sum();
   }

   /**
    * Calculates the number of false negatives for the given label
    *
    * @param label the label to calculate the number of false negatives of
    * @return the number of false negatives
    */
   public double falseNegatives(String label) {
      return matrix.get(label).sum() - matrix.get(label, label);
   }

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
    * Calculates the false positive rate of the given label
    *
    * @param label the label to calculate the false positive rate for
    * @return the false positive rate
    */
   public double falsePositiveRate(String label) {
      double tn = trueNegatives(label);
      double fp = falsePositives(label);
      if (tn + fp == 0) {
         return 0.0;
      }
      return fp / (tn + fp);
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
   public double falsePositives() {
      return matrix.firstKeys()
                   .stream()
                   .mapToDouble(k -> {
                                   double fp = 0;
                                   for (String o : matrix.firstKeys()) {
                                      if (!o.equals(k)) {
                                         fp += matrix.get(o, k);
                                      }
                                   }
                                   return fp;
                                }
                               ).sum();
   }

   /**
    * Calculates the number of false positives for the given label
    *
    * @param label the label to calculate the number of false positives for
    * @return the number of false positives
    */
   public double falsePositives(String label) {
      double fp = 0;
      for (String o : matrix.firstKeys()) {
         if (!o.equals(label)) {
            fp += matrix.get(o, label);
         }
      }
      return fp;
   }

   /**
    * Gets the set of labels in the classification
    *
    * @return the set of labels
    */
   public Set<String> labels() {
      return matrix.firstKeys();
   }

   /**
    * Calculates the macro F1-measure
    *
    * @return the macro F1-measure
    */
   public double macroF1() {
      return f1(macroPrecision(), macroRecall());
   }

   /**
    * Calculates the macro precision (average of all labels).
    *
    * @return the macro precision
    */
   public double macroPrecision() {
      return precisionPerClass().average();
   }

   /**
    * Calculates the macro recall (average of all labels).
    *
    * @return the macro recall
    */
   public double macroRecall() {
      return recallPerClass().average();
   }

   @Override
   public void merge(Evaluation<Instance, Classifier> evaluation) {
      if (evaluation != null) {
         Preconditions.checkArgument(evaluation instanceof ClassifierEvaluation,
                                     "Can only merge with other ClassifierEvaluation.");
         matrix.merge(Cast.<ClassifierEvaluation>as(evaluation).matrix);
         this.total += Cast.<ClassifierEvaluation>as(evaluation).total;
      }
   }

   /**
    * Calculates the micro F1-measure
    *
    * @return the micro F1-measure
    */
   public double microF1() {
      return f1(microPrecision(), microRecall());
   }

   /**
    * Calculates the micro precision.
    *
    * @return the micro precision
    */
   public double microPrecision() {
      double tp = truePositives();
      double fp = falsePositives();
      if (tp + fp == 0) {
         return 1.0;
      }
      return tp / (tp + fp);
   }

   /**
    * Calculates the micro recall.
    *
    * @return the micro recall
    */
   public double microRecall() {
      double tp = truePositives();
      double fn = falseNegatives();
      if (tp + fn == 0) {
         return 1.0;
      }
      return tp / (tp + fn);
   }

   /**
    * Calculates the negative likelihood ratio, which is <code>False Positive Rate / Specificity</code>
    *
    * @return the negative likelihood ratio
    */
   public double negativeLikelihoodRatio() {
      return falseNegativeRate() / specificity();
   }

   /**
    * Calculates the negative likelihood ratio of the given label
    *
    * @param label the label to calculate the negative likelihood ratio for
    * @return the negative likelihood ratio
    */
   public double negativeLikelihoodRatio(String label) {
      return falseNegativeRate(label) / specificity(label);
   }

   /**
    * Outputs the results of the classification as per-class Precision, Recall, and F1 and also includes the confusion
    * matrix.
    *
    * @param printStream the print stream to write to
    */
   @Override
   public void output(@NonNull PrintStream printStream) {
      output(printStream, true);
   }

   /**
    * Outputs the results of the classification as per-class Precision, Recall, and F1 and optionally the confusion
    * matrix.
    *
    * @param printStream          the print stream to write to
    * @param printConfusionMatrix True print the confusion matrix, False do not print the confusion matrix.
    */
   public void output(@NonNull PrintStream printStream, boolean printConfusionMatrix) {

      final Set<String> columns = matrix.entries().stream()
                                        .flatMap(e -> Stream.of(e.v1, e.v2))
                                        .distinct()
                                        .collect(Collectors.toCollection(TreeSet::new));

      Set<String> sorted = new TreeSet<>(matrix.firstKeys());

      TableFormatter tableFormatter = new TableFormatter();
      if (printConfusionMatrix) {
         tableFormatter.title("Confusion Matrix");
         tableFormatter.header(Collections.singleton(StringUtils.EMPTY));
         tableFormatter.header(columns);
         tableFormatter.header(Collections.singleton("Total"));
         sorted.forEach(gold -> {
            List<Object> row = new ArrayList<>();
            row.add(gold);
            columns.forEach(c -> row.add((long) matrix.get(gold, c)));
            row.add((long) matrix.get(gold).sum());
            tableFormatter.content(row);
         });
         List<Object> totalRow = new ArrayList<>();
         totalRow.add("Total");
         columns.forEach(c -> totalRow.add((long) matrix.firstKeys().stream()
                                                        .mapToDouble(k -> matrix.get(k, c))
                                                        .sum()));
         totalRow.add((long) matrix.sum());
         tableFormatter.content(totalRow);
         tableFormatter.print(printStream);
         printStream.println();
      }

      tableFormatter.clear();
      tableFormatter
         .title("Classification Metrics")
         .header(Arrays.asList(StringUtils.EMPTY, "Precision", "Recall", "F1-Measure", "Correct", "Incorrect", "Missed",
                               "Total"));

      sorted.forEach(g ->
                        tableFormatter.content(Arrays.asList(
                           g,
                           precision(g),
                           recall(g),
                           f1(g),
                           matrix.get(g, g),
                           falsePositives(g),
                           matrix.get(g).sum() - matrix.get(g, g),
                           matrix.get(g).sum()
                                                            ))
                    );
      tableFormatter.footer(Arrays.asList(
         "micro",
         microPrecision(),
         microRecall(),
         microF1(),
         truePositives(),
         falsePositives(),
         falseNegatives(),
         total
                                         ));
      tableFormatter.footer(Arrays.asList(
         "macro",
         macroPrecision(),
         macroRecall(),
         macroF1(),
         "-",
         "-",
         "-",
         "-"
                                         ));
      tableFormatter.print(printStream);

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
    * Calculates the positive likelihood ratio of the given label
    *
    * @param label the label to calculate the positive likelihood ratio for
    * @return the positive likelihood ratio
    */
   public double positiveLikelihoodRatio(String label) {
      return truePositiveRate(label) / falsePositiveRate(label);
   }

   /**
    * Calculates the precision of the given label, which is <code>True Positives / (True Positives + False
    * Positives)</code>
    *
    * @param label the label to calculate the precision of
    * @return the precision
    */
   public double precision(String label) {
      double tp = truePositives(label);
      double fp = falsePositives(label);
      if (tp + fp == 0) {
         return 1.0;
      }
      return tp / (tp + fp);
   }

   /**
    * Creates a counter where the items are labels and their values are their precision
    *
    * @return the counter of precision values
    */
   public Counter<String> precisionPerClass() {
      Counter<String> precisions = Counters.newCounter();
      matrix.firstKeys().forEach(k -> precisions.set(k, precision(k)));
      return precisions;
   }

   /**
    * Calculates the recall of the given label, which is <code>True Positives / (True Positives + True Negatives)</code>
    *
    * @param label the label to calculate the recall of
    * @return the recall
    */
   public double recall(String label) {
      double tp = truePositives(label);
      double fn = falseNegatives(label);
      if (tp + fn == 0) {
         return 1.0;
      }
      return tp / (tp + fn);
   }

   /**
    * Creates a counter where the items are labels and their values are their recall
    *
    * @return the counter of recall values
    */
   public Counter<String> recallPerClass() {
      Counter<String> recalls = Counters.newCounter();
      matrix.firstKeys().forEach(k -> recalls.set(k, recall(k)));
      return recalls;
   }

   /**
    * Calculates the sensitivity (same as the micro-averaged recall)
    *
    * @return the sensitivity
    */
   public double sensitivity() {
      return microRecall();
   }

   /**
    * Calculates the sensitivity of the given label (same as the micro-averaged recall)
    *
    * @param label the label to calculate the sensitivity of
    * @return the sensitivity
    */
   public double sensitivity(String label) {
      return recall(label);
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
    * Calculates the specificity of the given label
    *
    * @param label the label to calculate the specificity of
    * @return the specificity
    */
   public double specificity(String label) {
      double tn = trueNegatives(label);
      double fp = falsePositives(label);
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
    * Calculates the true negative rate (or specificity) of the given label
    *
    * @param label the label to calculate the true negative rate of
    * @return the true negative rate
    */
   public double trueNegativeRate(String label) {
      return specificity(label);
   }

   /**
    * Counts the number of true negatives
    *
    * @return the number of true negatives
    */
   public double trueNegatives() {
      return matrix.firstKeys()
                   .stream()
                   .mapToDouble(k -> {
                                   double tn = 0;
                                   for (String o : matrix.firstKeys()) {
                                      if (!o.equals(k)) {
                                         tn += matrix.get(o).sum() - matrix.get(o, k);
                                      }
                                   }
                                   return tn;
                                }
                               ).sum();
   }

   /**
    * Counts the number of true negatives for the given label
    *
    * @param label the label
    * @return the number of true negatvies
    */
   public double trueNegatives(String label) {
      double tn = 0;
      for (String o : matrix.firstKeys()) {
         if (!o.equals(label)) {
            tn += matrix.get(o).sum() - matrix.get(o, label);
         }
      }
      return tn;
   }

   /**
    * Calculates the true positive rate (same as micro recall).
    *
    * @return the true positive rate
    */
   public double truePositiveRate() {
      return microRecall();
   }

   /**
    * Calculates the true positive rate if the given label
    *
    * @param label the label
    * @return the true positive rate
    */
   public double truePositiveRate(String label) {
      return recall(label);
   }

   /**
    * Calculates the number of true positives.
    *
    * @return the number of true positive
    */
   public double truePositives() {
      return matrix.firstKeys().stream().mapToDouble(k -> matrix.get(k, k)).sum();
   }

   /**
    * Calculates the number of true positive for the given label
    *
    * @param label the label
    * @return the number of true positive
    */
   public double truePositives(String label) {
      return matrix.get(label, label);
   }

}//END OF ClassifierEvaluation
