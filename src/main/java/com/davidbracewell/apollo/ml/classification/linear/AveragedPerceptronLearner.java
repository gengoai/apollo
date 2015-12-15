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

package com.davidbracewell.apollo.ml.classification.linear;

import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.ml.Dataset;
import com.davidbracewell.apollo.ml.FeatureVector;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.classification.ClassifierLearner;
import com.davidbracewell.collection.Collect;
import com.davidbracewell.logging.Logger;
import com.google.common.base.Preconditions;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.text.DecimalFormat;

/**
 * @author David B. Bracewell
 */
public class AveragedPerceptronLearner extends ClassifierLearner {
  private static Logger log = Logger.getLogger(AveragedPerceptronLearner.class);
  private static final long serialVersionUID = 1L;
  private int maxIterations;
  private double learningRate;
  private Vector totalWeights;
  private Vector stamps;
  private double totalBias;
  private double biasStamps;
  private double tolerance;
  private boolean verbose = false;

  public AveragedPerceptronLearner() {
    this(100, 1.0, 0.0001);
  }

  public AveragedPerceptronLearner(int maxIterations, double learningRate, double tolerance) {
    this.maxIterations = maxIterations;
    this.learningRate = learningRate;
    this.tolerance = tolerance;
  }

  @Override
  protected BinaryGLM trainImpl(Dataset<Instance> dataset) {
    dataset.encode();
    Preconditions.checkArgument(dataset.getLabelEncoder().size() == 2, "Can only have two classes");
    BinaryGLM model = new BinaryGLM(
      dataset.getLabelEncoder(),
      dataset.getFeatureEncoder(),
      dataset.getPreprocessors()
    );

    totalWeights = new FeatureVector(model.getFeatureEncoder());
    stamps = new FeatureVector(model.getFeatureEncoder());
    model.weights = new FeatureVector(model.getFeatureEncoder());

    double c = 1d;
    double oldError = 0;
    double oldOldError = 0;
    final DecimalFormat formatter = new DecimalFormat("###.00%");
    for (int iteration = 0; iteration < maxIterations; iteration++) {
      double error = 0;
      double count = 0;
      for (Instance instance : dataset) {
        count++;
        FeatureVector v = instance.toVector(model.getFeatureEncoder(), model.getLabelEncoder());
        double y = v.getLabel();
        double yHat = model.classify(v).getEncodedResult();

        if (y != yHat) {
          error++;
          double eta = learningRate * (y - yHat);
          for (Vector.Entry entry : Collect.asIterable(v.nonZeroIterator())) {
            updateFeature(model,entry.getIndex(),c, eta);
//            model.weights.increment(entry.getIndex(), entry.getValue() * eta);
          }
//          model.bias += eta;
          double timeSpan = c - biasStamps;
          totalBias += (timeSpan * model.bias);
          model.bias += eta;
          biasStamps = c;
        }


        c++;
      }
      if (verbose) {
        log.info("iteration={0} errorRate={1}", iteration, formatter.format(error / count));
      }
      if (error == 0) {
        break;
      }
      error /= count;
      if (iteration > 2) {
        if (Math.abs(error - oldError) < tolerance && Math.abs(error - oldOldError) < tolerance) {
          break;
        }
      }
      oldOldError = oldError;
      oldError = error;
    }

    double time = c;
    for (Vector.Entry entry : Collect.asIterable(totalWeights.nonZeroIterator())) {
      double total = totalWeights.get(entry.index);
      total += (time - stamps.get(entry.index)) * model.weights.get(entry.index);
      total = new BigDecimal(total / time).setScale(3, RoundingMode.HALF_UP).doubleValue();
      model.weights.set(entry.index, total);
    }
    double total = totalBias;
    total += (time - biasStamps) * model.bias;
    total = new BigDecimal(total / time).setScale(3, RoundingMode.HALF_UP).doubleValue();
    model.bias = total;

    return model;
  }


  private void updateFeature(BinaryGLM model, int featureId, double time, double value) {
    double timeAtWeight = time - stamps.get(featureId);
    double curWeight = model.weights.get(featureId);
    totalWeights.increment(featureId, timeAtWeight * curWeight);
    stamps.set(featureId, time);
    model.weights.set(featureId, curWeight + value);
  }

  @Override
  public void reset() {
    this.totalBias = 0;
    this.biasStamps = 0;
    this.stamps = null;
    this.totalWeights = null;
  }

  public double getLearningRate() {
    return learningRate;
  }

  public void setLearningRate(double learningRate) {
    this.learningRate = learningRate;
  }

  public int getMaxIterations() {
    return maxIterations;
  }

  public void setMaxIterations(int maxIterations) {
    this.maxIterations = maxIterations;
  }

  public double getTolerance() {
    return tolerance;
  }

  public void setTolerance(double tolerance) {
    this.tolerance = tolerance;
  }
}//END OF AveragedPerceptronLearner
