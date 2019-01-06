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

package com.gengoai.apollo.ml.regression;

import com.gengoai.apollo.ml.Example;
import com.gengoai.apollo.ml.FitParameters;
import com.gengoai.apollo.ml.Split;
import com.gengoai.apollo.ml.data.Dataset;
import com.gengoai.logging.Logger;
import com.gengoai.stream.MStream;
import com.gengoai.string.TableFormatter;
import org.apache.mahout.math.list.DoubleArrayList;

import java.io.PrintStream;
import java.io.Serializable;
import java.util.Arrays;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * <p>Evaluation for regression models.</p>
 *
 * @author David B. Bracewell
 */
public class RegressionEvaluation implements Serializable {
   private static final Logger log = Logger.getLogger(RegressionEvaluation.class);
   private static final long serialVersionUID = 1L;
   private DoubleArrayList gold = new DoubleArrayList();
   private double p = 0;
   private DoubleArrayList predicted = new DoubleArrayList();


   /**
    * Cross validation multi class evaluation.
    *
    * @param dataset       the dataset
    * @param regression    the classifier
    * @param fitParameters the fit parameters
    * @param nFolds        the n folds
    * @return the multi class evaluation
    */
   public static RegressionEvaluation crossValidation(Dataset dataset,
                                                      Regression regression,
                                                      FitParameters fitParameters,
                                                      int nFolds
                                                     ) {
      RegressionEvaluation evaluation = new RegressionEvaluation();
      AtomicInteger foldId = new AtomicInteger(0);
      for (Split split : dataset.shuffle().fold(nFolds)) {
         if (fitParameters.verbose) {
            log.info("Running fold {0}", foldId.incrementAndGet());
         }
         regression.fit(split.train, fitParameters);
         evaluation.evaluate(regression, split.test);
         if (fitParameters.verbose) {
            log.info("Fold {0}: Cumulative Metrics(r2={1})", foldId.get(), evaluation.r2());
         }
      }
      return evaluation;
   }


   /**
    * Calculates the adjusted r2
    *
    * @return the adjusted r2
    */
   public double adjustedR2() {
      double r2 = r2();
      return r2 - (1.0 - r2) * p / (gold.size() - p - 1.0);
   }

   /**
    * Adds an entry to the evaluation
    *
    * @param gold      the gold value
    * @param predicted the predicted value
    */
   public void entry(double gold, double predicted) {
      this.gold.add(gold);
      this.predicted.add(predicted);
   }


   /**
    * Calculates the mean squared error
    *
    * @return the mean squared error
    */
   public double meanSquaredError() {
      return squaredError() / gold.size();
   }

   /**
    * Evaluate the given model using the given dataset
    *
    * @param model   the model to evaluate
    * @param dataset the dataset to evaluate over
    */
   public void evaluate(Regression model, Dataset dataset) {
      evaluate(model, dataset.stream());
   }

   public void evaluate(Regression model, MStream<Example> dataset) {
      for (Example ii : dataset) {
         gold.add(ii.getLabelAsDouble());
         predicted.add(model.estimate(ii));
      }
      p = Math.max(p, model.getNumberOfFeatures());
   }

   /**
    * Merge this evaluation with another combining the results.
    *
    * @param evaluation the other evaluation to combine
    */
   public void merge(RegressionEvaluation evaluation) {
      gold.addAllOf(evaluation.gold);
      predicted.addAllOf(evaluation.predicted);
   }


   /**
    * Outputs the evaluation results to the given print stream.
    *
    * @param printStream the print stream to write to
    */
   public void output(PrintStream printStream) {
      TableFormatter formatter = new TableFormatter();
      formatter.title("Regression Metrics");
      formatter.header(Arrays.asList("Metric", "Value"));
      formatter.content(Arrays.asList("RMSE", rootMeanSquaredError()));
      formatter.content(Arrays.asList("R^2", r2()));
      formatter.content(Arrays.asList("Adj. R^2", adjustedR2()));
      formatter.print(printStream);
   }

   /**
    * Outputs the evaluation results to standard out.
    */
   public final void output() {
      output(System.out);
   }

   /**
    * Calculates the r2
    *
    * @return the r2
    */
   public double r2() {
      double yMean = Arrays.stream(gold.elements())
                           .average().orElse(0d);
      double SSTO = Arrays.stream(gold.elements())
                          .map(d -> Math.pow(d - yMean, 2))
                          .sum();
      double SSE = squaredError();
      return 1.0 - (SSE / SSTO);
   }

   /**
    * Calculates the root mean squared error
    *
    * @return the root mean squared error
    */
   public double rootMeanSquaredError() {
      return Math.sqrt(meanSquaredError());
   }

   /**
    * Sets the total number of predictor variables (i.e. features)
    *
    * @param p the number of predictor variables
    */
   public void setP(double p) {
      this.p = p;
   }

   /**
    * Calculates the squared error
    *
    * @return the squared error
    */
   public double squaredError() {
      double error = 0;
      for (int i = 0; i < gold.size(); i++) {
         error += Math.pow(predicted.get(i) - gold.get(i), 2);
      }
      return error;
   }
}//END OF RegressionEvaluation
