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

import com.gengoai.Validation;
import com.gengoai.apollo.ml.Example;
import com.gengoai.collection.counter.Counter;
import com.gengoai.collection.counter.Counters;
import com.gengoai.collection.counter.MultiCounter;
import com.gengoai.collection.counter.MultiCounters;
import com.gengoai.conversion.Cast;
import com.gengoai.stream.MStream;
import com.gengoai.string.Strings;
import com.gengoai.string.TableFormatter;

import java.io.PrintStream;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static com.gengoai.tuple.Tuples.$;

/**
 * Provides various common metrics for measuring the quality of classifiers.
 *
 * @author David B. Bracewell
 */
public class MultiClassEvaluation extends ClassifierEvaluation {
   private static final long serialVersionUID = 1L;
   private final MultiCounter<String, String> confusionMatrix = MultiCounters.newMultiCounter();
   private double total = 0;


   @Override
   public double accuracy() {
      double correct = confusionMatrix.firstKeys().stream()
                                      .mapToDouble(k -> confusionMatrix.get(k, k))
                                      .sum();
      return correct / total;
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

   @Override
   public void entry(String gold, Classification predicted) {
      confusionMatrix.increment(gold, predicted.getResult());
      total++;
   }

   public void entry(String gold, String predicted) {
      confusionMatrix.increment(gold, predicted);
      total++;
   }

   @Override
   public void evaluate(Classifier model, MStream<Example> dataset) {
      dataset.filter(Example::hasLabel)
             .mapToPair(instance -> $(instance.getLabelAsString(), model.predict(instance)))
             .forEachLocal(this::entry);
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

   @Override
   public double falseNegatives() {
      return confusionMatrix.firstKeys()
                            .stream()
                            .mapToDouble(k -> confusionMatrix.get(k).sum() - confusionMatrix.get(k, k))
                            .sum();
   }

   /**
    * Calculates the number of false negatives for the given label
    *
    * @param label the label to calculate the number of false negatives of
    * @return the number of false negatives
    */
   public double falseNegatives(String label) {
      return confusionMatrix.get(label).sum() - confusionMatrix.get(label, label);
   }


   /**
    * Calculates the false positive rate of the given label
    *
    * @param label the label to calculate the false positive rate for
    * @return the false positive rate
    */
   public double falsePositiveRate(String label) {
      double tp = truePositives(label);
      double fn = falseNegatives(label);
      if (tp + fn == 0) {
         return 0.0;
      }
      return fn / (tp + fn);
   }

   @Override
   public double falsePositives() {
      return confusionMatrix.firstKeys()
                            .stream()
                            .mapToDouble(k -> {
                                            double fp = 0;
                                            for (String o : confusionMatrix.firstKeys()) {
                                               if (!o.equals(k)) {
                                                  fp += confusionMatrix.get(o, k);
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
      for (String o : confusionMatrix.firstKeys()) {
         if (!o.equals(label)) {
            fp += confusionMatrix.get(o, label);
         }
      }
      return fp;
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
      confusionMatrix.firstKeys().forEach(k -> f1.set(k, f1(p.get(k), r.get(k))));
      return f1;
   }

   /**
    * Creates a counter where the items are labels and their values are their precision
    *
    * @return the counter of precision values
    */
   public Counter<String> precisionPerClass() {
      Counter<String> precisions = Counters.newCounter();
      confusionMatrix.firstKeys().forEach(k -> precisions.set(k, precision(k)));
      return precisions;
   }

   /**
    * Creates a counter where the items are labels and their values are their recall
    *
    * @return the counter of recall values
    */
   public Counter<String> recallPerClass() {
      Counter<String> recalls = Counters.newCounter();
      confusionMatrix.firstKeys().forEach(k -> recalls.set(k, recall(k)));
      return recalls;
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
   public void merge(ClassifierEvaluation evaluation) {
      Validation.checkArgument(evaluation instanceof MultiClassEvaluation,
                               "Can only merge with other ClassifierEvaluation.");
      MultiClassEvaluation other = Cast.as(evaluation);
      confusionMatrix.merge(other.confusionMatrix);
      total += other.total;
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
    * Calculates the negative likelihood ratio of the given label
    *
    * @param label the label to calculate the negative likelihood ratio for
    * @return the negative likelihood ratio
    */
   public double negativeLikelihoodRatio(String label) {
      return falseNegativeRate(label) / specificity(label);
   }



   @Override
   public void output(PrintStream printStream, boolean printConfusionMatrix) {
      final Set<String> columns = confusionMatrix.entries().stream()
                                                 .flatMap(e -> Stream.of(e.v1, e.v2))
                                                 .collect(Collectors.toCollection(TreeSet::new));

      Set<String> sorted = new TreeSet<>(confusionMatrix.firstKeys());

      TableFormatter tableFormatter = new TableFormatter();
      if (printConfusionMatrix) {
         tableFormatter.title("Confusion Matrix");
         tableFormatter.header(Collections.singleton(Strings.EMPTY));
         tableFormatter.header(columns);
         tableFormatter.header(Collections.singleton("Total"));
         sorted.forEach(gold -> {
            List<Object> row = new ArrayList<>();
            row.add(gold);
            columns.forEach(c -> row.add((long) confusionMatrix.get(gold, c)));
            row.add((long) confusionMatrix.get(gold).sum());
            tableFormatter.content(row);
         });
         List<Object> totalRow = new ArrayList<>();
         totalRow.add("Total");
         columns.forEach(c -> totalRow.add((long) confusionMatrix.firstKeys().stream()
                                                                 .mapToDouble(k -> confusionMatrix.get(k, c))
                                                                 .sum()));
         totalRow.add((long) confusionMatrix.sum());
         tableFormatter.content(totalRow);
         tableFormatter.print(printStream);
         printStream.println();
      }

      tableFormatter.clear();
      tableFormatter
         .title("Classification Metrics")
         .header(
            Arrays.asList(Strings.EMPTY, "Precision", "Recall", "F1-Measure", "Correct", "Incorrect", "Missed",
                          "Total"));

      sorted.forEach(g -> tableFormatter.content(Arrays.asList(
         g,
         precision(g),
         recall(g),
         f1(g),
         (long) truePositives(g),
         (long) falsePositives(g),
         (long) falseNegatives(g),
         (long) confusionMatrix.get(g).sum())));

      tableFormatter.footer(Arrays.asList(
         "micro",
         microPrecision(),
         microRecall(),
         microF1(),
         (long) truePositives(),
         (long) falsePositives(),
         (long) falseNegatives(),
         (long) total));

      tableFormatter.footer(Arrays.asList(
         "macro",
         macroPrecision(),
         macroRecall(),
         macroF1(),
         "-",
         "-",
         "-",
         "-"));
      tableFormatter.print(printStream);

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
    * Calculates the recall of the given label, which is <code>True Positives / (True Positives + True
    * Negatives)</code>
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
    * Calculates the sensitivity of the given label (same as the micro-averaged recall)
    *
    * @param label the label to calculate the sensitivity of
    * @return the sensitivity
    */
   public double sensitivity(String label) {
      return recall(label);
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
    * Calculates the true negative rate (or specificity) of the given label
    *
    * @param label the label to calculate the true negative rate of
    * @return the true negative rate
    */
   public double trueNegativeRate(String label) {
      return specificity(label);
   }

   @Override
   public double trueNegatives() {
      return confusionMatrix.firstKeys()
                            .stream()
                            .mapToDouble(k -> {
                                            double tn = 0;
                                            for (String o : confusionMatrix.firstKeys()) {
                                               if (!o.equals(k)) {
                                                  tn += confusionMatrix.get(o).sum() - confusionMatrix.get(o, k);
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
      for (String o : confusionMatrix.firstKeys()) {
         if (!o.equals(label)) {
            tn += confusionMatrix.get(o).sum() - confusionMatrix.get(o, label);
         }
      }
      return tn;
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
      return confusionMatrix.firstKeys().stream().mapToDouble(k -> confusionMatrix.get(k, k)).sum();
   }

   /**
    * Calculates the number of true positive for the given label
    *
    * @param label the label
    * @return the number of true positive
    */
   public double truePositives(String label) {
      return confusionMatrix.get(label, label);
   }


}//END OF ClassifierEvaluation
