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
import com.gengoai.apollo.ml.vectorizer.DiscreteVectorizer;
import com.gengoai.conversion.Cast;
import com.gengoai.math.Math2;
import com.gengoai.stream.MStream;
import com.gengoai.string.TableFormatter;
import org.apache.mahout.math.list.DoubleArrayList;

import java.io.PrintStream;

import static java.util.Arrays.asList;

/**
 * <p>Common methods and metrics for evaluating a binary classifier Evaluation includes the following metrics in
 * addition to those in {@link ClassifierEvaluation}.</p>
 * <ul>
 * <li>Area Under the Curve (AUC)</li>
 * <li>Accuracy</li>
 * <li>False Positive Rate</li>
 * <li>False Negative Rate</li>
 * <li>True Positive Rate</li>
 * <li>True negative Rate</li>
 * <li>Majority Class Baseline</li>
 * </ul>
 *
 * @author David B. Bracewell
 */
public class BinaryEvaluation extends ClassifierEvaluation {
   private static final long serialVersionUID = 1L;
   private final String trueLabel;
   private final DoubleArrayList[] prob = {new DoubleArrayList(), new DoubleArrayList()};
   private double fn = 0;
   private double fp = 0;
   private double negative = 0d;
   private double positive = 0d;
   private double tn = 0;
   private double tp = 0;


   /**
    * Instantiates a new Binary evaluation.
    *
    * @param vectorizer the vectorizer
    */
   public BinaryEvaluation(DiscreteVectorizer vectorizer) {
      this(vectorizer.getString(1.0));
   }

   /**
    * Instantiates a new Binary evaluation.
    *
    * @param trueLabel the true label
    */
   public BinaryEvaluation(String trueLabel) {
      this.trueLabel = trueLabel;
   }


   @Override
   public double accuracy() {
      return (tp + tn) / (positive + negative);
   }

   /**
    * Calculates the AUC (Area Under the Curve)
    *
    * @return the AUC
    */
   public double auc() {
      return Math2.auc(prob[0].elements(),
                       prob[1].elements());
   }

   /**
    * Calculates the baseline score, which is <code>max(positive,negative) / (positive+negative)</code>
    *
    * @return the baseline score
    */
   public double baseline() {
      return Math.max(positive, negative) / (positive + negative);
   }

   @Override
   public void entry(String gold, Classification predicted) {
      int predictedClass = predicted.getResult().equals(trueLabel) ? 1 : 0;
      int goldClass = gold.equals(trueLabel) ? 1 : 0;
      prob[goldClass].add(predicted.getScore(trueLabel));
      if (goldClass == 1) {
         positive++;
         if (predictedClass == 1) {
            tp++;
         } else {
            fn++;
         }
      } else {
         negative++;
         if (predictedClass == 1) {
            fp++;
         } else {
            tn++;
         }
      }
   }

   @Override
   public void evaluate(Classifier model, MStream<Example> dataset) {
      dataset.forEach(instance -> entry(instance.getLabel(), model.predict(instance)));
   }

   @Override
   public double falseNegatives() {
      return fn;
   }

   @Override
   public double falsePositives() {
      return fp;
   }

   @Override
   public void merge(ClassifierEvaluation evaluation) {
      if (evaluation instanceof BinaryEvaluation) {
         BinaryEvaluation bce = Cast.as(evaluation);
         this.prob[0].addAllOf(bce.prob[0]);
         this.prob[1].addAllOf(bce.prob[1]);
      } else {
         throw new IllegalArgumentException();
      }
   }

   @Override
   public void output(PrintStream printStream, boolean printConfusionMatrix) {
      TableFormatter tableFormatter = new TableFormatter();

      if (printConfusionMatrix) {
         tableFormatter.header(asList("Predicted / Gold", "TRUE", "FALSE", "TOTAL"));
         tableFormatter.content(
            asList("TRUE", truePositives(), falsePositives(), (truePositives() + falsePositives())));
         tableFormatter.content(
            asList("FALSE", falseNegatives(), trueNegatives(), (falseNegatives() + trueNegatives())));
         tableFormatter.footer(asList("", (truePositives() + falseNegatives()), (falsePositives() + trueNegatives()),
                                      positive + negative));
         tableFormatter.print(printStream);
         tableFormatter = new TableFormatter();
      }

      tableFormatter.header(asList("Metric", "Score"));
      tableFormatter.content(asList("AUC", auc()));
      tableFormatter.content(asList("Accuracy", accuracy()));
      tableFormatter.content(asList("Baseline", baseline()));
      tableFormatter.content(asList("TP Rate", truePositiveRate()));
      tableFormatter.content(asList("FP Rate", falsePositiveRate()));
      tableFormatter.content(asList("TN Rate", trueNegativeRate()));
      tableFormatter.content(asList("FN Rate", falseNegativeRate()));
      tableFormatter.print(printStream);
   }

   private double toDouble(String string) {
      return string.equals(trueLabel) ? 1.0 : 0.0;
   }

   @Override
   public double trueNegatives() {
      return tn;
   }

   @Override
   public double truePositives() {
      return tp;
   }


}//END OF BinaryClassifierEvaluation
