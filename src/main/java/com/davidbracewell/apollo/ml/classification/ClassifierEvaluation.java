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

import com.davidbracewell.apollo.ml.Dataset;
import com.davidbracewell.apollo.ml.EncoderPair;
import com.davidbracewell.apollo.ml.Evaluation;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.collection.Counter;
import com.davidbracewell.collection.Counters;
import com.davidbracewell.collection.MultiCounter;
import com.davidbracewell.collection.MultiCounters;
import com.davidbracewell.conversion.Cast;
import com.davidbracewell.string.StringUtils;
import com.davidbracewell.tuple.Tuple2;
import com.google.common.base.Preconditions;
import lombok.NonNull;

import java.io.PrintStream;
import java.io.Serializable;
import java.text.DecimalFormat;
import java.util.Arrays;
import java.util.Collection;
import java.util.Set;
import java.util.TreeSet;
import java.util.stream.Collectors;

/**
 * The type Classifier evaluation.
 *
 * @author David B. Bracewell
 */
public class ClassifierEvaluation implements Evaluation<Instance, Classifier>, Serializable {
  private static final long serialVersionUID = 1L;
  private final MultiCounter<String, String> matrix = MultiCounters.newHashMapMultiCounter();
  private double total = 0;
  private EncoderPair encoderPair;

  /**
   * Instantiates a new Classifier evaluation.
   */
  public ClassifierEvaluation() {

  }

  /**
   * Instantiates a new Classifier evaluation.
   *
   * @param encoderPair the encoder pair
   */
  public ClassifierEvaluation(@NonNull EncoderPair encoderPair) {
    this.encoderPair = encoderPair;
  }

  /**
   * Accuracy double.
   *
   * @return the double
   */
  public double accuracy() {
    double correct = matrix.items().stream()
      .mapToDouble(k -> matrix.get(k, k))
      .sum();
    return correct / total;
  }

  @Override
  public void addEntry(double gold, double predicted) {
    Preconditions.checkNotNull(encoderPair, "ENCODER PAIR NOT SET");
    System.err.println(encoderPair.decodeLabel(gold));
    System.err.println(encoderPair.decodeLabel(predicted));
    matrix.increment(
      encoderPair.decodeLabel(gold).toString(),
      encoderPair.decodeLabel(predicted).toString()
    );
    total++;
  }

  /**
   * Add entry.
   *
   * @param gold      the gold
   * @param predicted the predicted
   */
  public void addEntry(String gold, String predicted) {
    matrix.increment(
      gold,
      predicted
    );
    total++;
  }

  /**
   * Diagnostic odds ratio double.
   *
   * @return the double
   */
  public double diagnosticOddsRatio() {
    return positiveLikelihoodRatio() / negativeLikelihoodRatio();
  }

  /**
   * Diagnostic odds ratio double.
   *
   * @param label the label
   * @return the double
   */
  public double diagnosticOddsRatio(String label) {
    return positiveLikelihoodRatio(label) / negativeLikelihoodRatio(label);
  }

  @Override
  public void evaluate(@NonNull Classifier model, @NonNull Dataset<Instance> dataset) {
    setEncoderPair(model.getEncoderPair());
    dataset.stream()
      .filter(Instance::hasLabel)
      .mapToPair(instance -> Tuple2.of(instance.getLabel().toString(), model.classify(instance).getResult()))
      .forEachLocal(this::addEntry);
  }

  @Override
  public void evaluate(@NonNull Classifier model, @NonNull Collection<Instance> dataset) {
    setEncoderPair(model.getEncoderPair());
    dataset.forEach(instance ->
      addEntry(
        model.encodeLabel(instance.getLabel()),
        model.classify(instance).getEncodedResult()
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
   * F 1 l per class counter.
   *
   * @return the counter
   */
  public Counter<String> f1PerClass() {
    Counter<String> f1 = Counters.newHashMapCounter();
    Counter<String> p = precisionPerClass();
    Counter<String> r = recallPerClass();
    matrix.items().forEach(k ->
      f1.set(k, f1(p.get(k), r.get(k)))
    );
    return f1;
  }

  /**
   * False negative rate double.
   *
   * @param label the label
   * @return the double
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
   * False negative rate double.
   *
   * @return the double
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
   * False negatives double.
   *
   * @return the double
   */
  public double falseNegatives() {
    return matrix.items().stream().mapToDouble(k -> matrix.get(k).sum() - matrix.get(k, k)).sum();
  }

  /**
   * False negatives double.
   *
   * @param label the label
   * @return the double
   */
  public double falseNegatives(String label) {
    return matrix.get(label).sum() - matrix.get(label, label);
  }

  /**
   * False positive rate double.
   *
   * @param label the label
   * @return the double
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
   * False positive rate double.
   *
   * @return the double
   */
  public double falsePositiveRate() {
    double tn = trueNegatives();
    double fp = falsePositives();
    return fp / (tn + fp);
  }

  /**
   * False positives double.
   *
   * @return the double
   */
  public double falsePositives() {
    return matrix.items()
      .stream()
      .mapToDouble(k -> {
          double fp = 0;
          for (String o : matrix.items()) {
            if (!o.equals(k)) {
              fp += matrix.get(o, k);
            }
          }
          return fp;
        }
      ).sum();
  }

  /**
   * False positives double.
   *
   * @param label the label
   * @return the double
   */
  public double falsePositives(String label) {
    double fp = 0;
    for (String o : matrix.items()) {
      if (!o.equals(label)) {
        fp += matrix.get(o, label);
      }
    }
    return fp;
  }

  public double falseOmissionRate() {
    double fn = falseNegatives();
    double tn = trueNegatives();
    return fn / (fn + tn);
  }

  /**
   * Gets encoder pair.
   *
   * @return the encoder pair
   */
  public EncoderPair getEncoderPair() {
    return encoderPair;
  }

  /**
   * Sets encoder pair.
   *
   * @param encoderPair the encoder pair
   */
  public void setEncoderPair(EncoderPair encoderPair) {
    this.encoderPair = encoderPair;
  }

  /**
   * Labels set.
   *
   * @return the set
   */
  public Set<String> labels() {
    return matrix.items();
  }

  /**
   * Macro f 1 double.
   *
   * @return the double
   */
  public double macroF1() {
    return f1(macroPrecision(), macroRecall());
  }

  /**
   * Macro precision double.
   *
   * @return the double
   */
  public double macroPrecision() {
    return precisionPerClass().average();
  }

  /**
   * Micro recall.
   *
   * @return the double
   */
  public double macroRecall() {
    return recallPerClass().average();
  }

  @Override
  public void merge(Evaluation<Instance, Classifier> evaluation) {
    if (evaluation != null) {
      Preconditions.checkArgument(evaluation instanceof ClassifierEvaluation, "Can only merge with other ClassifierEvaluation.");
      matrix.merge(Cast.<ClassifierEvaluation>as(evaluation).matrix);
      this.total += Cast.<ClassifierEvaluation>as(evaluation).total;
    }
  }

  /**
   * Micro f 1 double.
   *
   * @return the double
   */
  public double microF1() {
    return f1(microPrecision(), microRecall());
  }

  /**
   * Micro precision double.
   *
   * @return the double
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
   * Micro recall.
   *
   * @return the double
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
   * Negative likelihood ratio double.
   *
   * @return the double
   */
  public double negativeLikelihoodRatio() {
    return falseNegativeRate() / specificity();
  }

  /**
   * Negative likelihood ratio double.
   *
   * @param label the label
   * @return the double
   */
  public double negativeLikelihoodRatio(String label) {
    return falseNegativeRate(label) / specificity(label);
  }

  /**
   * Positive likelihood ratio double.
   *
   * @return the double
   */
  public double positiveLikelihoodRatio() {
    return microRecall() / falsePositiveRate();
  }

  /**
   * Positive likelihood ratio double.
   *
   * @param label the label
   * @return the double
   */
  public double positiveLikelihoodRatio(String label) {
    return recall(label) / falsePositiveRate(label);
  }

  public double positivePredictveValue() {
    return microPrecision();
  }

  public double positivePredictveValue(String label) {
    return precision(label);
  }

  public double precision(String label) {
    double tp = truePositives(label);
    double fp = falsePositives(label);
    if (tp + fp == 0) {
      return 1.0;
    }
    return tp / (tp + fp);
  }

  /**
   * Precision per class counter.
   *
   * @return the counter
   */
  public Counter<String> precisionPerClass() {
    Counter<String> precisions = Counters.newHashMapCounter();
    matrix.items().forEach(k -> precisions.set(k, precision(k)));
    return precisions;
  }

  public double recall(String label) {
    double tp = truePositives(label);
    double fn = falseNegatives(label);
    if (tp + fn == 0) {
      return 1.0;
    }
    return tp / (tp + fn);
  }

  /**
   * Recall per class counter.
   *
   * @return the counter
   */
  public Counter<String> recallPerClass() {
    Counter<String> recalls = Counters.newHashMapCounter();
    matrix.items().forEach(k -> recalls.set(k, recall(k)));
    return recalls;
  }

  /**
   * Sensitivity double.
   *
   * @return the double
   */
  public double sensitivity() {
    return microRecall();
  }

  /**
   * Sensitivity double.
   *
   * @param label the label
   * @return the double
   */
  public double sensitivity(String label) {
    return recall(label);
  }

  /**
   * Specificity double.
   *
   * @return the double
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
   * Specificity double.
   *
   * @param label the label
   * @return the double
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
   * True negatives double.
   *
   * @return the double
   */
  public double trueNegatives() {
    return matrix.items()
      .stream()
      .mapToDouble(k -> {
          double tn = 0;
          for (String o : matrix.items()) {
            if (!o.equals(k)) {
              tn += matrix.get(o).sum() - matrix.get(o, k);
            }
          }
          return tn;
        }
      ).sum();
  }

  /**
   * True negatives double.
   *
   * @param label the label
   * @return the double
   */
  public double trueNegatives(String label) {
    double tn = 0;
    for (String o : matrix.items()) {
      if (!o.equals(label)) {
        tn += matrix.get(o).sum() - matrix.get(o, label);
      }
    }
    return tn;
  }

  public double trueNegativeRate() {
    return specificity();
  }

  public double trueNegativeRate(String label) {
    return specificity(label);
  }

  public double truePositiveRate() {
    return microRecall();
  }

  public double truePositiveRate(String label) {
    return recall(label);
  }


  /**
   * True positives double.
   *
   * @return the double
   */
  public double truePositives() {
    return matrix.items().stream().mapToDouble(k -> matrix.get(k, k)).sum();
  }

  /**
   * True positives double.
   *
   * @param label the label
   * @return the double
   */
  public double truePositives(String label) {
    return matrix.get(label, label);
  }

  private String middleCMBar(String hbar, int nC) {
    StringBuilder builder = new StringBuilder();
    builder.append("├").append(hbar);
    for (int i = 1; i <= nC; i++) {
      builder.append("┼").append(hbar);
    }
    builder.append("┤");
    return builder.toString();
  }

  public void output(@NonNull PrintStream printStream) {
    final int longestLabel = Math.max(
      matrix.entries().stream().flatMap(e -> Arrays.asList(e.v1, e.v2).stream()).mapToInt(String::length).max().orElse(0) + 2,
      7
    );
    final Set<String> columns = matrix.entries().stream().map(e -> e.v2).distinct().collect(Collectors.toCollection(TreeSet::new));
    final DecimalFormat formatter = new DecimalFormat("####E");

    String horizontalBar = StringUtils.repeat("─", longestLabel);
    String blankColumn = StringUtils.repeat(' ', longestLabel);

    String hline = middleCMBar(horizontalBar, columns.size());


    StringBuilder builder = new StringBuilder();

    builder.append("┌").append(horizontalBar);
    for (int i = 1; i <= columns.size(); i++) {
      builder.append("┬").append(horizontalBar);
    }
    builder.append("┐");
    printStream.println(builder.toString());

    builder.setLength(0);
    builder.append("│").append(blankColumn);
    for (String column : columns) {
      builder.append("│").append(StringUtils.center(column, longestLabel));
    }
    builder.append("│");
    printStream.println(builder.toString());
    printStream.println(hline);

    int row = 0;
    for (String gold : new TreeSet<>(matrix.items())) {
      builder.setLength(0);
      builder.append("│").append(StringUtils.center(gold, longestLabel));
      for (String column : columns) {
        builder.append("│").append(StringUtils.center(formatter.format(matrix.get(gold, column)), longestLabel));
      }
      builder.append("│");
      printStream.println(builder.toString());
      row++;
      if (row < matrix.items().size()) {
        printStream.println(hline);
      }
    }

    builder.setLength(0);
    builder.append("└").append(horizontalBar);
    for (int i = 1; i <= columns.size(); i++) {
      builder.append("┴").append(horizontalBar);
    }
    builder.append("┘");
    printStream.println(builder.toString());


    printStream.println();

    printStream.println("       ┌───────┬───────┬───────┐");
    printStream.println("       │   P   │   R   │  F-1  │");
    printStream.println("┌──────├───────┼───────┼───────┤");
    printStream.printf("│micro │ %.3f │ %.3f │ %.3f │\n", microPrecision(), microRecall(), microF1());
    printStream.println("├──────┼───────┼───────┼───────┤");
    printStream.printf("│macro │ %.3f │ %.3f │ %.3f │\n", macroPrecision(), macroRecall(), macroF1());
    printStream.println("└──────┴───────┴───────┴───────┘\n");
  }


}//END OF ClassifierEvaluation
