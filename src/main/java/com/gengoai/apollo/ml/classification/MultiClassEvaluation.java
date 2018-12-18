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

package com.gengoai.apollo.ml.classification;

import com.gengoai.apollo.linear.Axis;
import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.linear.NDArrayFactory;
import com.gengoai.apollo.ml.Evaluation;
import com.gengoai.apollo.ml.FitParameters;
import com.gengoai.apollo.ml.Split;
import com.gengoai.apollo.ml.data.Dataset;
import com.gengoai.apollo.ml.vectorizer.IndexVectorizer;
import com.gengoai.apollo.ml.vectorizer.Vectorizer;
import com.gengoai.collection.counter.Counter;
import com.gengoai.collection.counter.Counters;
import com.gengoai.conversion.Cast;
import com.gengoai.logging.Logger;
import com.gengoai.math.Math2;
import com.gengoai.string.Strings;
import com.gengoai.string.TableFormatter;

import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Consumer;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static com.gengoai.Validation.checkArgument;
import static com.gengoai.Validation.checkState;

/**
 * Provides various common metrics for measuring the quality of classifiers.
 *
 * @author David B. Bracewell
 */
public class MultiClassEvaluation implements ClassifierEvaluation {
   private static final Logger log = Logger.getLogger(MultiClassEvaluation.class);
   private static final long serialVersionUID = 1L;
   private final NDArray confusionMatrix;
   private final int numberOfLabels;
   private Vectorizer<String> labelVectorizer = null;
   private double total = 0;

   /**
    * Instantiates a new Multi class evaluation.
    *
    * @param classifier the classifier
    */
   public MultiClassEvaluation(PipelinedClassifier classifier) {
      this(classifier.getLabelVectorizer());
   }

   /**
    * Instantiates a new Multi class evaluation.
    *
    * @param numberOfLabels the number of labels
    */
   public MultiClassEvaluation(int numberOfLabels) {
      this.numberOfLabels = numberOfLabels;
      this.confusionMatrix = NDArrayFactory.SPARSE.zeros(numberOfLabels, numberOfLabels);
   }

   /**
    * Instantiates a new Multi class evaluation.
    *
    * @param vectorizer the vectorizer
    */
   public MultiClassEvaluation(Vectorizer<String> vectorizer) {
      this.labelVectorizer = vectorizer;
      this.numberOfLabels = vectorizer.size();
      this.confusionMatrix = NDArrayFactory.SPARSE.zeros(numberOfLabels, numberOfLabels);
   }

   @Override
   public double accuracy() {
      double correct = 0;
      for (int i = 0; i < numberOfLabels; i++) {
         correct += confusionMatrix.get(i, i);
      }
      return correct / total;
   }

   public static MultiClassEvaluation crossValidation(Dataset dataset,
                                                      PipelinedClassifier classifier,
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
   public static MultiClassEvaluation crossValidation(Dataset dataset,
                                                      PipelinedClassifier classifier,
                                                      FitParameters fitParameters,
                                                      int nFolds
                                                     ) {

      IndexVectorizer vectorizer = IndexVectorizer.labelVectorizer();
      vectorizer.fit(dataset);

      MultiClassEvaluation evaluation = new MultiClassEvaluation(vectorizer);
      AtomicInteger foldId = new AtomicInteger(0);
      for (Split split : dataset.fold(nFolds)) {
         if (fitParameters.verbose) {
            log.info("Running fold {0}", foldId.incrementAndGet());
         }
         classifier.fit(split.train, fitParameters);
         classifier.evaluate(split.test, evaluation);
         if (fitParameters.verbose) {
            log.info("Fold {0}: Cumulative Metrics(microP={1}, microR={2}, microF1={3})", foldId.get(),
                     evaluation.microPrecision(),
                     evaluation.microRecall(),
                     evaluation.microF1());
         }
      }
      return evaluation;
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
      entry(labelVectorizer.encode(gold),
            labelVectorizer.encode(predicted.getResult()));
   }

   @Override
   public void entry(double gold, double predicted) {
      confusionMatrix.increment((int) gold, (int) predicted, 1.0);
      total++;
   }

   @Override
   public void entry(NDArray entry) {
      entry(getMax(entry.getLabelAsNDArray()),
            getMax(entry.getPredictedAsNDArray()));
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
    * Calculates the F1-measure for the given label, which is calculated as <code>(2 * precision(label) * recall(label))
    * / (precision(label) + recall(label)</code>
    *
    * @param label the label to calculate the F1 measure for
    * @return the f1 measure
    */
   public double f1(double label) {
      return f1(precision(label), recall(label));
   }

   /**
    * Calculates the F1 measure for each class
    *
    * @return a Counter where the items are labels and the values are F1 scores
    */
   public Counter<String> labeledF1PerClass() {
      Counter<String> f1 = Counters.newCounter();
      for (int i = 0; i < numberOfLabels; i++) {
         f1.set(labelVectorizer.decode(i), f1(i));
      }
      return f1;
   }

   /**
    * Calculates the F1 measure for each class
    *
    * @return a Counter where the items are labels and the values are F1 scores
    */
   public double[] f1PerClass() {
      double[] f1 = new double[numberOfLabels];
      for (int i = 0; i < numberOfLabels; i++) {
         f1[i] = f1(i);
      }
      return f1;
   }

   /**
    * Calculates the false negative rate of the given label
    *
    * @param label the label to calculate the false negative rate of
    * @return the false negative rate
    */
   public double falseNegativeRate(String label) {
      checkState(labelVectorizer != null, "A label vectorizer is not set.");
      return falseNegativeRate(labelVectorizer.encode(label));
   }

   /**
    * Calculates the false negative rate of the given label
    *
    * @param label the label to calculate the false negative rate of
    * @return the false negative rate
    */
   public double falseNegativeRate(double label) {
      double tp = truePositives(label);
      double fn = falseNegatives(label);
      if (tp + fn == 0) {
         return 0.0;
      }
      return fn / (tp + fn);
   }

   @Override
   public double falseNegatives() {
      double sum = 0;
      for (int i = 0; i < numberOfLabels; i++) {
         sum += falseNegatives(i);
      }
      return sum;
   }


   /**
    * Calculates the number of false negatives for the given label
    *
    * @param label the label to calculate the number of false negatives of
    * @return the number of false negatives
    */
   public double falseNegatives(String label) {
      checkState(labelVectorizer != null, "A label vectorizer is not set.");
      return falseNegatives(labelVectorizer.encode(label));
   }


   /**
    * Calculates the number of false negatives for the given label
    *
    * @param label the label to calculate the number of false negatives of
    * @return the number of false negatives
    */
   public double falseNegatives(double label) {
      return confusionMatrix.getVector((int) label, Axis.ROW).scalarSum() - confusionMatrix.get((int) label,
                                                                                                (int) label);
   }

   /**
    * Calculates the false positive rate of the given label
    *
    * @param label the label to calculate the false positive rate for
    * @return the false positive rate
    */
   public double falsePositiveRate(String label) {
      checkState(labelVectorizer != null, "A label vectorizer is not set.");
      return falsePositiveRate(labelVectorizer.encode(label));
   }

   /**
    * Calculates the false positive rate of the given label
    *
    * @param label the label to calculate the false positive rate for
    * @return the false positive rate
    */
   public double falsePositiveRate(double label) {
      double tn = trueNegatives(label);
      double fp = falsePositives(label);
      if (tn + fp == 0) {
         return 0.0;
      }
      return fp / (tn + fp);
   }

   @Override
   public double falsePositives() {
      double fp = 0;
      for (int i = 0; i < numberOfLabels; i++) {
         fp += falsePositives(i);
      }
      return fp;
   }

   /**
    * Calculates the number of false positives for the given label
    *
    * @param label the label to calculate the number of false positives for
    * @return the number of false positives
    */
   public double falsePositives(String label) {
      checkState(labelVectorizer != null, "A label vectorizer is not set.");
      return falsePositives(labelVectorizer.encode(label));
   }

   /**
    * Calculates the number of false positives for the given label
    *
    * @param label the label to calculate the number of false positives for
    * @return the number of false positives
    */
   public double falsePositives(double label) {
      double fp = 0;
      for (int i = 0; i < numberOfLabels; i++) {
         if (i != label) {
            fp += confusionMatrix.get(i, (int) label);
         }
      }
      return fp;
   }

   private int getMax(NDArray array) {
      if (array.isScalar()) {
         return (int) array.scalarValue();
      }
      return (int) array.argMax(Axis.ROW).get(0);
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
      return Math2.sum(precisionPerClass()) / numberOfLabels;
   }

   /**
    * Calculates the macro recall (average of all labels).
    *
    * @return the macro recall
    */
   public double macroRecall() {
      return Math2.sum(recallPerClass()) / numberOfLabels;
   }

   @Override
   public void merge(Evaluation evaluation) {
      checkArgument(evaluation instanceof MultiClassEvaluation,
                    "Can only merge with other ClassifierEvaluation.");
      MultiClassEvaluation mce = Cast.as(evaluation);
      this.confusionMatrix.addi(mce.confusionMatrix);
      this.total += mce.total;
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

   /**
    * Calculates the negative likelihood ratio of the given label
    *
    * @param label the label to calculate the negative likelihood ratio for
    * @return the negative likelihood ratio
    */
   public double negativeLikelihoodRatio(double label) {
      return falseNegativeRate(label) / specificity(label);
   }


   /**
    * Outputs the results of the classification as per-class Precision, Recall, and F1 and also includes the confusion
    * matrix.
    *
    * @param printStream the print stream to write to
    */
   @Override
   public void output(PrintStream printStream) {
      output(printStream, true);
   }


   private String toLabel(int i) {
      return labelVectorizer == null
             ? Integer.toString(i)
             : labelVectorizer.decode(i);
   }

   /**
    * Outputs the results of the classification as per-class Precision, Recall, and F1 and optionally the confusion
    * matrix.
    *
    * @param printStream          the print stream to write to
    * @param printConfusionMatrix True print the confusion matrix, False do not print the confusion matrix.
    */
   public void output(PrintStream printStream, boolean printConfusionMatrix) {
      NDArray rowSums = confusionMatrix.sum(Axis.ROW);
      NDArray colSums = confusionMatrix.sum(Axis.COLUMN);

      TableFormatter tableFormatter = new TableFormatter();
      if (printConfusionMatrix) {
         tableFormatter.title("Confusion Matrix");
         tableFormatter.header(Collections.singleton(Strings.EMPTY));
         tableFormatter.header(IntStream.range(0, numberOfLabels).mapToObj(this::toLabel).collect(Collectors.toList()));
         tableFormatter.header(Collections.singleton("Total"));
         for (int i = 0; i < numberOfLabels; i++) {
            List<Object> row = new ArrayList<>();
            row.add(toLabel(i));
            confusionMatrix.rowIterator(i)
                           .forEachRemaining(e -> row.add((long) e.getValue()));
            row.add((long) rowSums.get(i));
            tableFormatter.content(row);
         }
         List<Object> totalRow = new ArrayList<>();
         totalRow.add("Total");
         for (int i = 0; i < numberOfLabels; i++) {
            totalRow.add((long) colSums.get(i));
         }
         totalRow.add((long) confusionMatrix.scalarSum());
         tableFormatter.content(totalRow);
         tableFormatter.print(printStream);
         printStream.println();
      }

      tableFormatter.clear();
      tableFormatter
         .title("Classification Metrics")
         .header(Arrays.asList(Strings.EMPTY, "Precision", "Recall", "F1-Measure", "Correct", "Incorrect", "Missed",
                               "Total"));

      for (int g = 0; g < numberOfLabels; g++) {
         tableFormatter.content(Arrays.asList(
            toLabel(g),
            precision(g),
            recall(g),
            f1(g),
            (long) confusionMatrix.get(g, g),
            (long) falsePositives(g),
            (long) rowSums.get(g) - (long) confusionMatrix.get(g, g),
            (long) rowSums.get(g)));
      }

      tableFormatter.footer(Arrays.asList(
         "micro",
         microPrecision(),
         microRecall(),
         microF1(),
         (long) truePositives(),
         (long) falsePositives(),
         (long) falseNegatives(),
         (long) total
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
    * Calculates the positive likelihood ratio of the given label
    *
    * @param label the label to calculate the positive likelihood ratio for
    * @return the positive likelihood ratio
    */
   public double positiveLikelihoodRatio(String label) {
      return truePositiveRate(label) / falsePositiveRate(label);
   }

   /**
    * Calculates the positive likelihood ratio of the given label
    *
    * @param label the label to calculate the positive likelihood ratio for
    * @return the positive likelihood ratio
    */
   public double positiveLikelihoodRatio(double label) {
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
      checkState(labelVectorizer != null, "A label vectorizer is not set.");
      return precision(labelVectorizer.encode(label));
   }

   /**
    * Calculates the precision of the given label, which is <code>True Positives / (True Positives + False
    * Positives)</code>
    *
    * @param label the label to calculate the precision of
    * @return the precision
    */
   public double precision(double label) {
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
   public Counter<String> labeledPrecisionPerClass() {
      checkState(labelVectorizer != null, "A label vectorizer is not set.");
      Counter<String> precisions = Counters.newCounter();
      for (int i = 0; i < numberOfLabels; i++) {
         precisions.set(labelVectorizer.decode(i), precision(i));
      }
      return precisions;
   }

   /**
    * Creates a counter where the items are labels and their values are their precision
    *
    * @return the counter of precision values
    */
   public double[] precisionPerClass() {
      double[] precisions = new double[numberOfLabels];
      for (int i = 0; i < numberOfLabels; i++) {
         precisions[i] = precision(i);
      }
      return precisions;
   }

   /**
    * Calculates the recall of the given label, which is <code>True Positives / (True Positives + True
    * Negatives)</code>
    *
    * @param label the label to calculate the recall of
    * @return the recall
    */
   public double recall(String label) {
      checkState(labelVectorizer != null, "A label vectorizer is not set.");
      return recall(labelVectorizer.encode(label));
   }

   /**
    * Calculates the recall of the given label, which is <code>True Positives / (True Positives + True
    * Negatives)</code>
    *
    * @param label the label to calculate the recall of
    * @return the recall
    */
   public double recall(double label) {
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
   public Counter<String> labeledRecallPerClass() {
      checkState(labelVectorizer != null, "A label vectorizer is not set.");
      Counter<String> recalls = Counters.newCounter();
      for (int i = 0; i < numberOfLabels; i++) {
         recalls.set(labelVectorizer.decode(i), recall(i));
      }
      return recalls;
   }

   /**
    * Creates a counter where the items are labels and their values are their recall
    *
    * @return the counter of recall values
    */
   public double[] recallPerClass() {
      double[] recalls = new double[numberOfLabels];
      for (int i = 0; i < numberOfLabels; i++) {
         recalls[i] = recall(i);
      }
      return recalls;
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
    * Calculates the sensitivity of the given label (same as the micro-averaged recall)
    *
    * @param label the label to calculate the sensitivity of
    * @return the sensitivity
    */
   public double sensitivity(double label) {
      return recall(label);
   }

   /**
    * Calculates the specificity of the given label
    *
    * @param label the label to calculate the specificity of
    * @return the specificity
    */
   public double specificity(String label) {
      checkState(labelVectorizer != null, "A label vectorizer is not set.");
      return specificity(labelVectorizer.encode(label));
   }

   /**
    * Calculates the specificity of the given label
    *
    * @param label the label to calculate the specificity of
    * @return the specificity
    */
   public double specificity(double label) {
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

   /**
    * Calculates the true negative rate (or specificity) of the given label
    *
    * @param label the label to calculate the true negative rate of
    * @return the true negative rate
    */
   public double trueNegativeRate(double label) {
      return specificity(label);
   }

   @Override
   public double trueNegatives() {
      double sum = 0;
      for (int i = 0; i < numberOfLabels; i++) {
         sum += trueNegatives(i);
      }
      return sum;
   }

   /**
    * Counts the number of true negatives for the given label
    *
    * @param label the label
    * @return the number of true negatvies
    */
   public double trueNegatives(String label) {
      checkState(labelVectorizer != null, "A label vectorizer is not set.");
      return trueNegatives(labelVectorizer.encode(label));
   }

   /**
    * Counts the number of true negatives for the given label
    *
    * @param label the label
    * @return the number of true negatvies
    */
   public double trueNegatives(double label) {
      double tn = 0;
      for (int i = 0; i < numberOfLabels; i++) {
         if (i != label) {
            NDArray row = confusionMatrix.getVector(i, Axis.ROW);
            tn += row.scalarSum() - confusionMatrix.get(i, (int) label);
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
    * Calculates the true positive rate if the given label
    *
    * @param label the label
    * @return the true positive rate
    */
   public double truePositiveRate(double label) {
      return recall(label);
   }

   /**
    * Calculates the number of true positives.
    *
    * @return the number of true positive
    */
   public double truePositives() {
      double sum = 0;
      for (int i = 0; i < numberOfLabels; i++) {
         sum += confusionMatrix.get(i, i);
      }
      return sum;
   }

   /**
    * Calculates the number of true positive for the given label
    *
    * @param label the label
    * @return the number of true positive
    */
   public double truePositives(String label) {
      checkState(labelVectorizer != null, "A label vectorizer is not set.");
      return truePositives(labelVectorizer.encode(label));
   }

   /**
    * Calculates the number of true positive for the given label
    *
    * @param label the label
    * @return the number of true positive
    */
   public double truePositives(double label) {
      return confusionMatrix.get((int) label, (int) label);
   }


}//END OF ClassifierEvaluation
