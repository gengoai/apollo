package com.davidbracewell.apollo.ml.regression;

import com.davidbracewell.apollo.ml.Dataset;
import com.davidbracewell.apollo.ml.Evaluation;
import com.davidbracewell.apollo.ml.FeatureVector;
import com.davidbracewell.apollo.ml.Instance;
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
 * @author David B. Bracewell
 */
public class RegressionEvaluation implements Evaluation<Instance, Regression> {
  DoubleArrayList gold = new DoubleArrayList();
  DoubleArrayList predicted = new DoubleArrayList();
  private double p = 0;

  public static void main(String[] args) {
    RegressionEvaluation re = new RegressionEvaluation();
    for (int i = 0; i < 100; i++) {
      double g = Math.random();
      double p = g + (0.5 * g) * Math.random();
      re.entry(g, p);
    }
    re.setP(5);
    re.output(System.out);
  }

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

  public double squaredError() {
    double error = 0;
    for (int i = 0; i < gold.size(); i++) {
      error += Math.pow(predicted.get(i) - gold.get(i), 2);
    }
    return error;
  }

  public double meanSquaredError() {
    return squaredError() / gold.size();
  }

  public double rootMeanSquaredError() {
    return Math.sqrt(meanSquaredError());
  }

  public double adjustedR2() {
    double r2 = r2();
    return r2 - (1.0 - r2) * p / (gold.size() - p - 1.0);
  }

  public double r2() {
    double yMean = DoubleStream.of(gold.elements()).sum() / gold.size();
    double SStot = DoubleStream.of(gold.elements()).map(d -> Math.pow(d - yMean, 2)).sum();
    double SSres = 0;
    for (int i = 0; i < gold.size(); i++) {
      SSres += Math.pow(gold.get(i) - predicted.get(i), 2);
    }
    return 1.0 - SSres / SStot;
  }

  public void entry(double predicted, double gold) {
    this.gold.add(gold);
    this.predicted.add(predicted);
  }

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
