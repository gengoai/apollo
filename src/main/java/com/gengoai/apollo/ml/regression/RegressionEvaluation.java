package com.gengoai.apollo.ml.regression;

import com.gengoai.Validation;
import com.gengoai.apollo.ml.Evaluation;
import com.gengoai.apollo.ml.Example;
import com.gengoai.apollo.ml.FitParameters;
import com.gengoai.apollo.ml.Split;
import com.gengoai.apollo.ml.data.Dataset;
import com.gengoai.conversion.Cast;
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
public class RegressionEvaluation implements Evaluation<Regression>, Serializable {
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

   @Override
   public void evaluate(Regression model, MStream<Example> dataset) {
      for (Example ii : dataset) {
         gold.add(ii.getLabelAsDouble());
         predicted.add(model.estimate(ii));
      }
      p = Math.max(p, model.getNumberOfFeatures());
   }

   @Override
   public void merge(Evaluation evaluation) {
      Validation.checkArgument(evaluation instanceof RegressionEvaluation);
      RegressionEvaluation re = Cast.as(evaluation);
      gold.addAllOf(re.gold);
      predicted.addAllOf(re.predicted);
   }


   @Override
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
