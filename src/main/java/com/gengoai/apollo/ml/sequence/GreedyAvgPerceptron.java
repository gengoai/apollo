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

package com.gengoai.apollo.ml.sequence;

import com.gengoai.Stopwatch;
import com.gengoai.apollo.ml.Example;
import com.gengoai.apollo.ml.Feature;
import com.gengoai.apollo.ml.FitParameters;
import com.gengoai.apollo.ml.ModelParameters;
import com.gengoai.apollo.ml.data.Dataset;
import com.gengoai.apollo.ml.preprocess.Preprocessor;
import com.gengoai.apollo.ml.preprocess.PreprocessorList;
import com.gengoai.apollo.ml.vectorizer.IndexVectorizer;
import com.gengoai.apollo.ml.vectorizer.NoOptVectorizer;
import com.gengoai.apollo.optimization.TerminationCriteria;
import com.gengoai.collection.HashBasedTable;
import com.gengoai.collection.Iterables;
import com.gengoai.collection.Table;
import com.gengoai.collection.counter.Counter;
import com.gengoai.collection.counter.Counters;
import com.gengoai.collection.counter.MultiCounter;
import com.gengoai.collection.counter.MultiCounters;
import com.gengoai.conversion.Cast;
import com.gengoai.logging.Loggable;

import java.util.Collections;
import java.util.HashSet;
import java.util.Set;

import static com.gengoai.Validation.notNull;

/**
 * The type Greedy avg perceptron.
 *
 * @author David B. Bracewell
 */
public class GreedyAvgPerceptron extends SequenceLabeler implements Loggable {
   private static final long serialVersionUID = 1L;
   private static final Feature BIAS_FEATURE = Feature.booleanFeature("******BIAS******");
   private Set<String> labelSet;
   private MultiCounter<String, String> featureWeights;
   private MultiCounter<String, String> transitionWeights;

   public GreedyAvgPerceptron(Preprocessor... preprocessors) {
      this(new PreprocessorList(preprocessors));
   }

   public GreedyAvgPerceptron(Validator validator, Preprocessor... preprocessors) {
      this(validator, new PreprocessorList(preprocessors));
   }


   public GreedyAvgPerceptron(PreprocessorList preprocessors) {
      this(Validator.ALWAYS_TRUE, preprocessors);
   }

   public GreedyAvgPerceptron(Validator validator, PreprocessorList preprocessors) {
      super(ModelParameters.create(new NoOptVectorizer<>())
                           .update(p -> {
                              p.preprocessors(preprocessors);
                              p.sequenceValidator = Validator.ALWAYS_TRUE;
                              p.featureVectorizer = new NoOptVectorizer<>();
                              p.sequenceValidator = validator;
                           }));
   }


   private Iterable<Feature> expandFeatures(Example example) {
      return Iterables.concat(example.getFeatures(), Collections.singleton(BIAS_FEATURE));
   }

   private Counter<String> distribution(Example example, String pLabel) {
      Counter<String> scores = Counters.newCounter(transitionWeights.get(pLabel));
      for (Feature feature : expandFeatures(example)) {
         scores.merge(featureWeights.get(feature.name).adjustValues(v -> v * feature.value));
      }
      return scores;
   }

   @Override
   protected SequenceLabeler fitPreprocessed(Dataset preprocessed, FitParameters fitParameters) {
      Parameters parameters = notNull(Cast.as(fitParameters, Parameters.class));

      IndexVectorizer vectorizer = IndexVectorizer.labelVectorizer();
      vectorizer.fit(preprocessed);
      this.labelSet = new HashSet<>(vectorizer.alphabet());


      this.featureWeights = MultiCounters.newMultiCounter();
      this.transitionWeights = MultiCounters.newMultiCounter();
      final MultiCounter<String, String> fTotals = MultiCounters.newMultiCounter();
      final MultiCounter<String, String> tTotals = MultiCounters.newMultiCounter();
      final Table<String, String, Integer> fTimestamps = new HashBasedTable<>();
      final Table<String, String, Integer> tTimestamps = new HashBasedTable<>();

      int instances = 0;
      TerminationCriteria terminationCriteria = TerminationCriteria.create()
                                                                   .historySize(parameters.historySize)
                                                                   .maxIterations(parameters.maxIterations)
                                                                   .tolerance(parameters.eps);

      for (int i = 0; i < terminationCriteria.maxIterations(); i++) {
         Stopwatch sw = Stopwatch.createStarted();
         double total = 0;
         double correct = 0;

         for (Example sequence : preprocessed.shuffle().stream()) {
            String pLabel = "<BOS>";
            for (int j = 0; j < sequence.size(); j++) {
               total++;
               instances++;
               Example instance = sequence.getExample(j);
               String y = instance.getLabelAsString();
               String predicted = distribution(instance, pLabel).max();

               if (predicted == null) {
                  predicted = vectorizer.decode(0);
               }

               if (!y.equals(predicted)) {
                  for (Feature feature : expandFeatures(instance)) {
                     update(y, feature.name, 1.0, instances, featureWeights, fTimestamps, fTotals);
                     update(predicted, feature.name, -1.0, instances, featureWeights, fTimestamps, fTotals);
                  }
                  update(y, pLabel, 1.0, instances, transitionWeights, fTimestamps, fTotals);
                  update(predicted, pLabel, -1.0, instances, transitionWeights, fTimestamps, fTotals);
               } else {
                  correct++;
               }
               pLabel = y;
            }
         }
         double error = 1d - (correct / total);

         sw.stop();
         if (parameters.verbose) {
            logInfo("Iteration {0}: Accuracy={1,number,#.####}, time to complete={2}", i + 1, (1d - error), sw);
         }

         if (terminationCriteria.check(error)) {
            break;
         }
      }

      average(instances, featureWeights, fTimestamps, fTotals);
      average(instances, transitionWeights, tTimestamps, tTotals);
      return this;
   }

   @Override
   public Parameters getDefaultFitParameters() {
      return new Parameters();
   }

   @Override
   public int getNumberOfFeatures() {
      return featureWeights.size();
   }

   @Override
   public int getNumberOfLabels() {
      return labelSet.size();
   }

   @Override
   public Labeling label(Example example) {
      String[] labels = new String[example.size()];
      String pLabel = "<BOS>";
      for (int i = 0; i < example.size(); i++) {
         Example instance = preprocess(example.getExample(i));
         Counter<String> distribution = distribution(instance, pLabel);
         String cLabel = distribution.max();
         distribution.remove(cLabel);
         while (!isValidTransition(cLabel, pLabel, instance)) {
            cLabel = distribution.max();
            distribution.remove(cLabel);
         }
         labels[i] = cLabel;
         pLabel = cLabel;
      }
      return new Labeling(labels);
   }

   private void update(String cls, String feature, double value, int iteration,
                       MultiCounter<String, String> weights,
                       Table<String, String, Integer> timeStamp,
                       MultiCounter<String, String> totals
                      ) {
      int iterAt = iteration - timeStamp.getOrDefault(feature, cls, 0);
      totals.increment(feature, cls, iterAt * weights.get(feature, cls));
      weights.increment(feature, cls, value);
      timeStamp.put(feature, cls, iteration);
   }

   private void average(int finalIteration,
                        MultiCounter<String, String> weights,
                        Table<String, String, Integer> timeStamp,
                        MultiCounter<String, String> totals
                       ) {
      for (String feature : new HashSet<>(weights.firstKeys())) {
         Counter<String> newWeights = Counters.newCounter();
         weights.get(feature)
                .forEach((cls, value) -> {
                   double total = totals.get(feature, cls);
                   total += (finalIteration - timeStamp.getOrDefault(feature, cls, 0)) * value;
                   double v = total / finalIteration;
                   if (Math.abs(v) >= 0.001) {
                      newWeights.set(cls, v);
                   }
                });
         weights.set(feature, newWeights);
      }
   }


   /**
    * Custom fit parameters for the GreedyAveragePerceptron
    */
   public static class Parameters extends FitParameters {
      private static final long serialVersionUID = 1L;
      /**
       * The epsilon to use for checking for convergence.
       */
      public double eps = 1e-4;
      /**
       * The number of iterations to use for determining convergence
       */
      public int historySize = 3;
      /**
       * The maximum number of iterations to run for
       */
      public int maxIterations = 100;
   }


}//END OF WindowSequenceLabeler
