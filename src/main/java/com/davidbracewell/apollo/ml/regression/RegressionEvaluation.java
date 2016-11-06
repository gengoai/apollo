package com.davidbracewell.apollo.ml.regression;

import com.davidbracewell.apollo.ml.Evaluation;
import com.davidbracewell.apollo.ml.FeatureVector;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.data.Dataset;
import com.davidbracewell.conversion.Cast;
import com.davidbracewell.string.TableFormatter;
import com.google.common.base.Preconditions;
import lombok.NonNull;
import org.apache.mahout.math.list.DoubleArrayList;

import java.io.PrintStream;
import java.util.Arrays;
import java.util.Collection;
import java.util.stream.DoubleStream;

/**
 * The type Regression evaluation.
 *
 * @author David B. Bracewell
 */
public class RegressionEvaluation implements Evaluation<Instance, Regression> {
   private DoubleArrayList gold = new DoubleArrayList();
   private DoubleArrayList predicted = new DoubleArrayList();
   private double p = 0;


   @Override
   public void evaluate(@NonNull Regression model, @NonNull Dataset<Instance> dataset) {
      for (Instance ii : dataset) {
         FeatureVector fv = ii.toVector(model.getEncoderPair());
         gold.add(fv.getLabel());
         predicted.add(model.estimate(fv));
      }
      p = model.numberOfFeatures();
   }

   @Override
   public void evaluate(@NonNull Regression model, @NonNull Collection<Instance> dataset) {
      for (Instance ii : dataset) {
         FeatureVector fv = ii.toVector(model.getEncoderPair());
         gold.add(fv.getLabel());
         predicted.add(model.estimate(fv));
      }
      p = model.numberOfFeatures();
   }

   @Override
   public void merge(@NonNull Evaluation<Instance, Regression> evaluation) {
      Preconditions.checkArgument(evaluation instanceof RegressionEvaluation);
      RegressionEvaluation re = Cast.as(evaluation);
      gold.addAllOf(re.gold);
      predicted.addAllOf(re.predicted);
   }

   /**
    * Squared error double.
    *
    * @return the double
    */
   public double squaredError() {
      double error = 0;
      for (int i = 0; i < gold.size(); i++) {
         error += Math.pow(predicted.get(i) - gold.get(i), 2);
      }
      return error;
   }

   /**
    * Mean squared error double.
    *
    * @return the double
    */
   public double meanSquaredError() {
      return squaredError() / gold.size();
   }

   /**
    * Root mean squared error double.
    *
    * @return the double
    */
   public double rootMeanSquaredError() {
      return Math.sqrt(meanSquaredError());
   }

   /**
    * Adjusted r 2 double.
    *
    * @return the double
    */
   public double adjustedR2() {
      double r2 = r2();
      return r2 - (1.0 - r2) * p / (gold.size() - p - 1.0);
   }

   /**
    * R 2 double.
    *
    * @return the double
    */
   public double r2() {
      double yMean = DoubleStream.of(gold.elements()).parallel().sum() / gold.size();
      double SStot = DoubleStream.of(gold.elements()).parallel().map(d -> Math.pow(d - yMean, 2)).sum();
      double SSres = 0;
      for (int i = 0; i < gold.size(); i++) {
         SSres += Math.pow(gold.get(i) - predicted.get(i), 2);
      }
      return 1.0 - SSres / SStot;
   }

   /**
    * Entry.
    *
    * @param predicted the predicted
    * @param gold      the gold
    */
   public void entry(double predicted, double gold) {
      this.gold.add(gold);
      this.predicted.add(predicted);
   }

   /**
    * Sets p.
    *
    * @param p the p
    */
   public void setP(double p) {
      this.p = p;
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

}// END OF RegressionEvaluation
