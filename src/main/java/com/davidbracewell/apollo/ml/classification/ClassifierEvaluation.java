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
import com.davidbracewell.collection.counter.HashMapCounter;
import com.davidbracewell.collection.counter.HashMapMultiCounter;
import com.davidbracewell.collection.counter.MultiCounter;
import com.davidbracewell.conversion.Cast;
import com.davidbracewell.logging.Logger;
import com.davidbracewell.string.StringUtils;
import com.davidbracewell.string.TableFormatter;
import com.davidbracewell.tuple.Tuple2;
import com.google.common.base.Preconditions;
import lombok.NonNull;

import java.io.PrintStream;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Supplier;
import java.util.stream.Collectors;

/**
 * The type Classifier evaluation.
 *
 * @author David B. Bracewell
 */
public class ClassifierEvaluation implements Evaluation<Instance, Classifier>, Serializable {
  private static final long serialVersionUID = 1L;
  private static final Logger log = Logger.getLogger(ClassifierEvaluation.class);
  private final MultiCounter<String, String> matrix = new HashMapMultiCounter<>();
  private double total = 0;

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


  /**
   * Add entry.
   *
   * @param gold      the gold
   * @param predicted the predicted
   */
  public void entry(String gold, String predicted) {
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
   * F 1 l per class counter.
   *
   * @return the counter
   */
  public Counter<String> f1PerClass() {
    Counter<String> f1 = new HashMapCounter<>();
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

  /**
   * False omission rate double.
   *
   * @return the double
   */
  public double falseOmissionRate() {
    double fn = falseNegatives();
    double tn = trueNegatives();
    return fn / (fn + tn);
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

  /**
   * Positive predictve value double.
   *
   * @return the double
   */
  public double positivePredictveValue() {
    return microPrecision();
  }

  /**
   * Positive predictve value double.
   *
   * @param label the label
   * @return the double
   */
  public double positivePredictveValue(String label) {
    return precision(label);
  }

  /**
   * Precision double.
   *
   * @param label the label
   * @return the double
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
   * Precision per class counter.
   *
   * @return the counter
   */
  public Counter<String> precisionPerClass() {
    Counter<String> precisions = new HashMapCounter<>();
    matrix.items().forEach(k -> precisions.set(k, precision(k)));
    return precisions;
  }

  /**
   * Recall double.
   *
   * @param label the label
   * @return the double
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
   * Recall per class counter.
   *
   * @return the counter
   */
  public Counter<String> recallPerClass() {
    Counter<String> recalls = new HashMapCounter<>();
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

  /**
   * True negative rate double.
   *
   * @return the double
   */
  public double trueNegativeRate() {
    return specificity();
  }

  /**
   * True negative rate double.
   *
   * @param label the label
   * @return the double
   */
  public double trueNegativeRate(String label) {
    return specificity(label);
  }

  /**
   * True positive rate double.
   *
   * @return the double
   */
  public double truePositiveRate() {
    return microRecall();
  }

  /**
   * True positive rate double.
   *
   * @param label the label
   * @return the double
   */
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

  /**
   * Output.
   *
   * @param printStream the print stream
   */
  @Override
  public void output(@NonNull PrintStream printStream) {
    output(printStream, true);
  }

  public double f1(String label) {
    return f1(precision(label), recall(label));
  }

  /**
   * Output.
   *
   * @param printStream          the print stream
   * @param printConfusionMatrix the print confusion matrix
   */
  public void output(@NonNull PrintStream printStream, boolean printConfusionMatrix) {

    final Set<String> columns = matrix.entries().stream()
      .flatMap(e -> Arrays.asList(e.v1, e.v2).stream())
      .distinct()
      .collect(Collectors.toCollection(TreeSet::new));

    Set<String> sorted = new TreeSet<>(matrix.items());

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
      columns.forEach(c -> {
        totalRow.add((long) matrix.items().stream()
          .mapToDouble(k -> matrix.get(k, c))
          .sum());
      });
      totalRow.add((long) matrix.sum());
      tableFormatter.content(totalRow);
      tableFormatter.print(printStream);
      printStream.println();
    }

    tableFormatter.clear();
    tableFormatter
      .title("Classification Metrics")
      .header(Arrays.asList(StringUtils.EMPTY, "Precision", "Recall", "F1-Measure", "Correct", "Incorrect", "Missed", "Total"));

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
    tableFormatter.content(Arrays.asList(
      "micro",
      microPrecision(),
      microRecall(),
      microF1(),
      truePositives(),
      falsePositives(),
      falseNegatives(),
      total
    ));
    tableFormatter.content(Arrays.asList(
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


  public ClassifierEvaluation crossValidation(@NonNull Dataset<Instance> dataset, @NonNull Supplier<ClassifierLearner> learnerSupplier, int nFolds) {
    AtomicInteger foldId = new AtomicInteger(0);
    dataset.fold(nFolds).forEach((train, test) -> {
      log.info("Running fold {0}", foldId.incrementAndGet());
      Classifier model = learnerSupplier.get().train(train);
      evaluate(model, test);
      log.info("Fold {0}: Cumulative Metrics(microP={1}, microR={2}, microF1={3})", foldId.get(), microPrecision(), microRecall(), microF1());
    });
    return this;
  }

}//END OF ClassifierEvaluation
