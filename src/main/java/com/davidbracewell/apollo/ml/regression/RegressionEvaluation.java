package com.davidbracewell.apollo.ml.regression;

import com.davidbracewell.Math2;
import com.davidbracewell.apollo.ml.Evaluation;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.data.Dataset;
import com.davidbracewell.conversion.Cast;
import com.davidbracewell.guava.common.base.Preconditions;
import com.davidbracewell.string.TableFormatter;
import lombok.NonNull;
import org.apache.mahout.math.list.DoubleArrayList;

import java.io.PrintStream;
import java.util.Arrays;
import java.util.Collection;

/**
 * <p>Evaluation for regression models.</p>
 *
 * @author David B. Bracewell
 */
public class RegressionEvaluation implements Evaluation<Instance, Regression> {
   private DoubleArrayList gold = new DoubleArrayList();
   private DoubleArrayList predicted = new DoubleArrayList();
   private double p = 0;

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

   @Override
   public void evaluate(@NonNull Regression model, @NonNull Dataset<Instance> dataset) {
      for (Instance ii : dataset) {
         gold.add(model.getLabelEncoder().encode(ii.getLabel()));
         predicted.add(model.estimate(ii));
      }
      p = model.numberOfFeatures();
   }

   @Override
   public void evaluate(@NonNull Regression model, @NonNull Collection<Instance> dataset) {
      for (Instance ii : dataset) {
         gold.add(model.getLabelEncoder().encode(ii.getLabel()));
         predicted.add(model.estimate(ii));
      }
      p = model.numberOfFeatures();
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
   public void merge(@NonNull Evaluation<Instance, Regression> evaluation) {
      Preconditions.checkArgument(evaluation instanceof RegressionEvaluation);
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
      double yMean = Math2.sum(gold.elements()) / gold.size();
      double SStot = Arrays.stream(gold.elements()).map(d -> Math.pow(d - yMean, 2)).sum();
      double SSres = 0;
      for (int i = 0; i < gold.size(); i++) {
         SSres += Math.pow(gold.get(i) - predicted.get(i), 2);
      }
      return 1.0 - SSres / SStot;
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

}// END OF RegressionEvaluation
