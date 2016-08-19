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

package com.davidbracewell.apollo.ml.sequence.linear;

import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.ml.data.Dataset;
import com.davidbracewell.apollo.ml.Feature;
import com.davidbracewell.apollo.ml.FeatureVector;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.sequence.ContextualIterator;
import com.davidbracewell.apollo.ml.sequence.Sequence;
import com.davidbracewell.apollo.ml.sequence.SequenceLabeler;
import com.davidbracewell.apollo.ml.sequence.SequenceLabelerLearner;
import com.davidbracewell.apollo.ml.sequence.decoder.BeamDecoder;
import com.davidbracewell.collection.Collect;
import lombok.Getter;
import lombok.Setter;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

import static com.davidbracewell.collection.CollectionHelpers.asStream;

/**
 * The type Crf learner.
 *
 * @author David B. Bracewell
 */
public class CRFLearner extends SequenceLabelerLearner {
  private static final long serialVersionUID = 1L;
  private double lambda = 0.0001;
  private double t = 1;
  public static final double ETA_0 = 0.1;
  public static final double FACTOR = 2.;
  public static final double BEST_ETA_INIT = 1.;
  public static final int MAX_SAMPLE_SZ = 1000;

  @Setter
  @Getter
  private int maxIterations = 100;
  @Setter
  @Getter
  private double eta = 1.0;

  private double c = 1.0;
  private double kEta = 0;
  @Setter
  @Getter
  private boolean verbose = true;

  private CRF model = null;
  private int numLabels = 0;

  void init(List<Sequence> trainingExamples) {
    double s0 = System.currentTimeMillis();
    t = trainingExamples.size();
    lambda = 1.0 / (c * t);
    if (kEta != 0.0) {
      t = 1.0 / (kEta * lambda);
    } else {
      // Otherwise find it
      initSchedule(trainingExamples.subList(0, Math.min(MAX_SAMPLE_SZ, trainingExamples.size())), ETA_0);
    }

    double sNow = System.currentTimeMillis();

    System.err.println("Initialized in " + (sNow - s0) / 1000. + "s");
  }

  private double findObjBySampling(List<Sequence> sample, CRF clone) {
    double loss = 0.;
    int sSz = sample.size();
    double wnorm = model.mag();
    for (int i = 0; i < sSz; ++i) {
      Scorer scorer = new Scorer(clone, sample.get(i));
      double forward = scorer.computeForward();
      double correct = scorer.computeCorrect();
      loss += forward - correct;
    }
    return loss / sSz + 0.5 * wnorm * lambda;
  }

  private double tryEtaBySampling(List<Sequence> sample, CRF clone, double eta) {
    int sSz = sample.size();
    for (int i = 0; i < sSz; ++i) {
      Scorer scorer = new Scorer(clone, sample.get(i));
      scorer.gradCorrect(1, eta);
      scorer.gradForward(-1, eta);
      clone.scale *= (1 - eta * lambda);
    }
    return findObjBySampling(sample, clone);
  }

  private void initSchedule(List<Sequence> sample, double eta0) {
    double obj0 = findObjBySampling(sample, model);

    System.err.println("Initial objective=" + obj0);
    double bestEta = BEST_ETA_INIT;
    double bestObj = obj0;
    double etaGuess = eta0;

    boolean phase2 = false;

    for (int k = 10; k > 0 || !phase2; ) {
      CRF clone = this.model.copy();
      double obj = tryEtaBySampling(sample, clone, etaGuess);
      boolean ok = (obj < obj0);

      System.err.println("Trying eta=" + etaGuess + " obj=" + obj);
      if (ok) {
        System.err.println(" (possible)");
      } else {
        System.err.println("(too large)");
      }
      if (ok) {
        --k;
        if (obj < bestObj) {
          bestObj = obj;

          bestEta = etaGuess;
        }
      }
      if (!phase2) {
        if (ok) {
          etaGuess *= FACTOR;
        } else {
          phase2 = true;
          etaGuess = eta0;
        }
      }
      if (phase2) {
        etaGuess /= FACTOR;
      }

    }
    bestEta /= FACTOR;
    t = 1.0 / (bestEta * lambda);


  }


  @Override
  protected SequenceLabeler trainImpl(Dataset<Sequence> dataset) {
    model = new CRF(
      dataset.getLabelEncoder(),
      dataset.getFeatureEncoder(),
      dataset.getPreprocessors(),
      getTransitionFeatures(),
      getValidator()
    );
    model.setDecoder(new BeamDecoder(100));


    numLabels = model.numberOfLabels();

    model.scale = 1;
    model.weights = new Vector[numLabels];
    for (int i = 0; i < numLabels; i++) {
      model.weights[i] = new FeatureVector(model.getEncoderPair());
    }
    List<Sequence> sequences = dataset.stream().collect();

    init(sequences);
    for (int iteration = 1; iteration <= maxIterations; iteration++) {
      trainIteration(iteration, sequences);
    }
    model.rescale();
    return model;
  }

  protected void trainIteration(int iteration, List<Sequence> dataset) {
    Collections.shuffle(dataset);
    for (Sequence sequence : dataset) {
      double eta = 1.0 / (lambda * t);
      Scorer scorer = new Scorer(this.model, sequence);
      scorer.gradCorrect(1, eta);
      scorer.gradForward(-1, eta);
      model.scale *= (1 - eta * lambda);
      t++;
    }

    if (model.scale <= 1e-5) {
      model.rescale();
    }

    if (verbose) {
      System.err.println("Iteration [" + iteration + "] wnorm=" + model.mag());
    }
  }

  @Override
  public void reset() {
    model = null;
    t = 0;
    numLabels = 0;
  }


  private static class Scorer {
    private double[][] u;
    final Sequence sequence;
    final CRF model;
    final int numLabels;

    private Scorer(CRF model, Sequence sequence) {
      this.model = model;
      this.sequence = sequence;
      this.numLabels = model.getLabelEncoder().size();
      compute();
    }


    void updateU(double[] g, int y, int labelStop, double eta, Vector vector) {
      double gain = eta / model.scale;
      vector.forEachSparse(offset -> {
        for (int k = 0; k < labelStop; k++) {
          model.weights[y].increment(offset.index, offset.value * g[k] * gain);
        }
      });
    }

    public double computeForward() {
      ContextualIterator<Instance> iterator = sequence.iterator();
      iterator.next();
      double[] scores = new double[numLabels];
      System.arraycopy(u[0], 0, scores, 0, numLabels);
      while (iterator.hasNext()) {
        iterator.next();
        int pos = iterator.getIndex();
        for (int i = 0; i < numLabels; i++) {
          scores[i] += u[pos][i];
        }
      }
      return logSum(scores);
    }


    double computeCorrect() {
      ContextualIterator<Instance> iterator = sequence.iterator();
      iterator.next();
      int y = (int) model.encodeLabel(iterator.getLabel());
      double sum = u[0][y];
      while (iterator.hasNext()) {
        iterator.next();
        y = (int) model.encodeLabel(iterator.getLabel());
        sum += u[iterator.getIndex()][y];
      }
      return sum;
    }

    double logSum(double[] v) {
      double m = v[0];
      for (int i = 1; i < v.length; ++i) {
        m = Math.max(m, v[i]);
      }
      double s = 0.;
      for (int i = 0; i < v.length; ++i) {
        s += Math.exp(-(m - v[i]));
      }
      return m + Math.log(s);
    }

    void dLogSum(double g, double[] v, double[] r) {
      double m = v[0];
      for (int i = 0; i < v.length; ++i) {
        m = Math.max(m, v[i]);
      }
      double z = 0.;
      for (int i = 0; i < v.length; ++i) {
        r[i] = Math.exp(-(m - v[i]));
        z += r[i];
      }
      for (int i = 0; i < v.length; ++i) {
        r[i] *= g / z;
      }
    }

    double gradForward(double g, double eta) {

      double[][] scores = new double[sequence.size()][numLabels];
      double[] uAcc = new double[numLabels];
      System.arraycopy(u[0], 0, scores[0], 0, numLabels);

      List<Vector> vectors = new ArrayList<>();
      for (ContextualIterator<Instance> iterator = sequence.iterator(); iterator.hasNext(); ) {
        iterator.next();
        vectors.add(toVector(iterator));
        int pos = iterator.getIndex();
        for (int i = 0; i < numLabels; i++) {
          scores[pos][i] += u[pos][i];
        }
      }

      double score = logSum(scores[sequence.size() - 1]);
      double[] grads = new double[numLabels];
      dLogSum(g, scores[sequence.size() - 1], grads);
      for (int pos = vectors.size() - 1; pos > 0; pos--) {
        Arrays.fill(uAcc, 0);
        updateU(grads, 0, numLabels, eta, vectors.get(pos));
        System.arraycopy(uAcc, 0, grads, 0, numLabels);
      }
      updateU(grads, 0, numLabels, eta, vectors.get(0));
      return score;
    }


    double gradCorrect(double g, double eta) {
      ContextualIterator<Instance> iterator = sequence.iterator();
      iterator.next();

      int y = (int) model.encodeLabel(iterator.getLabel());
      double[] grad = new double[]{g};
      updateU(grad, y, 1, eta, toVector(iterator));
      double sum = u[0][y];

      while (iterator.hasNext()) {
        iterator.next();
        int pos = iterator.getIndex();
        y = (int) model.encodeLabel(iterator.getLabel());
        sum += u[pos][y];
        updateU(grad, y, 1, eta, toVector(iterator));
      }

      return sum;
    }

    private Vector toVector(ContextualIterator<Instance> iterator) {
      return Instance.create(
        Collect.union(
          iterator.getCurrent().getFeatures(),
          asStream(model.getTransitionFeatures().extract(iterator)).map(Feature::TRUE).collect(Collectors.toList())
        )
      ).toVector(model.getEncoderPair());
    }

    private void compute() {
      u = new double[sequence.size()][numLabels];
      for (ContextualIterator<Instance> iterator = sequence.iterator(); iterator.hasNext(); ) {
        iterator.next();
        final int pos = iterator.getIndex();
        final Vector v = toVector(iterator);
        for (int y = 0; y < numLabels; y++) {
          u[pos][y] = model.weights[y].dot(v);
        }
        if (model.scale != 1) {
          for (int y = 0; y < numLabels; y++) {
            u[pos][y] *= model.scale;
          }
        }

      }
    }
  }


}//END OF CRFLearner
